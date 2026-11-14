use super::test_db::MvccTestDbNoConn;
use std::num::NonZeroU64;
use turso_core::alloc::vec;
use turso_core::mvcc::database::checkpoint::*;
use turso_core::mvcc::database::MVTableId;
use turso_core::mvcc::database::MvStore;
use turso_core::mvcc::database::Row;
use turso_core::mvcc::database::RowID;
use turso_core::mvcc::database::RowKey;
use turso_core::mvcc::database::RowVersion;
use turso_core::mvcc::database::TxTimestampOrID;
use turso_core::sync::Arc;
use turso_core::sync::RwLock;
use turso_core::types::ImmutableRecord;
use turso_core::CheckpointMode;
use turso_core::Value;

fn committed_table_row_version(table_id: MVTableId, rowid: i64) -> (RowID, RowVersion) {
    table_row_version(table_id, rowid, 1, Some(5), None, false)
}

fn table_row_version(
    table_id: MVTableId,
    rowid: i64,
    version_id: u64,
    begin: Option<u64>,
    end: Option<u64>,
    btree_resident: bool,
) -> (RowID, RowVersion) {
    let record = ImmutableRecord::from_values(&[Value::from_i64(rowid)], 1).unwrap();
    let row =
        Row::new_table_row(RowID::new(table_id, RowKey::Int(rowid)), record.as_blob()).unwrap();
    let version = RowVersion {
        id: version_id,
        begin: turso_core::mvcc::database::PackedTs::pack(begin.map(TxTimestampOrID::Timestamp)),
        end: turso_core::mvcc::database::PackedTs::pack(end.map(TxTimestampOrID::Timestamp)),
        data: row.data,
        btree_resident,
        materialized_at: turso_core::mvcc::database::WalPos::ORIGIN,
    };
    (row.id, version)
}

fn checkpoint_for_collect_tests(
) -> CheckpointStateMachine<turso_core::mvcc::clock::MvccClock, turso_core::alloc::DynAllocator> {
    let db = MvccTestDbNoConn::new();
    let conn = db.connect();
    let mvstore = db.get_mvcc_store();
    let pager = conn.pager.load().clone();
    let mut checkpoint = CheckpointStateMachine::new(
        pager,
        mvstore,
        conn.clone(),
        true,
        conn.get_sync_mode(),
        turso_core::MAIN_DB_ID,
        CheckpointMode::Truncate {
            upper_bound_inclusive: None,
        },
    );
    checkpoint.durable_txid_max_old = NonZeroU64::new(2);
    checkpoint
}

fn insert_row_version(
    mvstore: &turso_core::sync::Arc<
        MvStore<turso_core::mvcc::clock::MvccClock, turso_core::alloc::DynAllocator>,
    >,
    (key, version): (RowID, RowVersion),
) {
    let mut versions =
        <turso_core::mvcc::database::RowVersionChain<turso_core::alloc::DynAllocator> as turso_core::alloc::TursoVecInExt<
            RowVersion,
            turso_core::alloc::DynAllocator,
        >>::new_in(turso_core::alloc::DynAllocator::default());
    let stamp = version_stamp(&version);
    versions.push(version);
    mvstore
        .rows
        .insert(key.clone(), Arc::new(RwLock::new(versions)));
    mvstore.mark_checkpoint_dirty_key(&key, stamp).unwrap();
}

fn version_stamp(version: &RowVersion) -> u64 {
    [version.begin(), version.end()]
        .into_iter()
        .filter_map(|stamp| match stamp {
            Some(TxTimestampOrID::Timestamp(ts)) => Some(ts),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn insert_clean_row_version(
    mvstore: &turso_core::sync::Arc<
        MvStore<turso_core::mvcc::clock::MvccClock, turso_core::alloc::DynAllocator>,
    >,
    (key, version): (RowID, RowVersion),
) {
    let mut versions =
        <turso_core::mvcc::database::RowVersionChain<turso_core::alloc::DynAllocator> as turso_core::alloc::TursoVecInExt<
            RowVersion,
            turso_core::alloc::DynAllocator,
        >>::new_in(turso_core::alloc::DynAllocator::default());
    versions.push(version);
    mvstore.rows.insert(key, Arc::new(RwLock::new(versions)));
}

fn dirty_table_key(
    mvstore: &turso_core::sync::Arc<
        MvStore<turso_core::mvcc::clock::MvccClock, turso_core::alloc::DynAllocator>,
    >,
    table_id: MVTableId,
    rowid: i64,
) -> bool {
    mvstore
        .checkpoint_dirty_table_keys
        .contains_key(&RowID::new(table_id, RowKey::Int(rowid)))
}

#[test]
fn collect_table_rows_tolerates_dirty_key_without_chain() {
    let mut checkpoint = checkpoint_for_collect_tests();
    let mvstore = checkpoint.mvstore.clone();
    let table_id = MVTableId::from(-2);
    mvstore
        .mark_checkpoint_dirty_key(&RowID::new(table_id, RowKey::Int(7)), 5)
        .unwrap();
    checkpoint.snapshot_ts = 10;

    while checkpoint.collect_table_rows().unwrap().is_some() {}
    assert!(checkpoint.write_set.is_empty());

    assert!(checkpoint.prune_dirty_keys().is_none());
    assert!(!dirty_table_key(&mvstore, table_id, 7));
}

#[test]
fn dirty_walk_finds_near_and_far_keys_and_skips_missing_chains() {
    let mut checkpoint = checkpoint_for_collect_tests();
    let mvstore = checkpoint.mvstore.clone();
    let table_id = MVTableId::from(-2);
    for rowid in 1..=200 {
        insert_clean_row_version(
            &mvstore,
            table_row_version(table_id, rowid, 1, Some(1), None, false),
        );
    }
    for rowid in [1, 2, 3, 150, 200] {
        insert_row_version(&mvstore, committed_table_row_version(table_id, rowid));
    }
    mvstore
        .mark_checkpoint_dirty_key(&RowID::new(table_id, RowKey::Int(60)), 5)
        .unwrap();
    mvstore.rows.remove(&RowID::new(table_id, RowKey::Int(60)));

    while checkpoint.collect_table_rows().unwrap().is_some() {}

    let collected: std::vec::Vec<i64> = checkpoint
        .write_set
        .iter()
        .map(|(row_id, _, _)| row_id.row_id.to_int_or_panic())
        .collect();
    assert_eq!(collected, vec![1, 2, 3, 150, 200]);
}
