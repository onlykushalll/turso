use rustc_hash::FxHashSet as HashSet;

use super::*;
use crate::alloc::{TursoAllocator, TursoIteratorExt, TursoTryWithCapacityExt};
use crate::io::IO;
use crate::mvcc::clock::MvccClock;
#[cfg(feature = "conn_raw_api")]
use crate::mvcc::portable_logical::{PortableLogicalBuilder, PortableObjectMapEntry};
use crate::storage::sqlite3_ondisk::DatabaseHeader;
use crate::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::sync::Mutex;
use crate::sync::RwLock;
use crate::{LimboError, OpenFlags};
use quickcheck::{Arbitrary, Gen};
use quickcheck_macros::quickcheck;
use std::sync::Arc;

#[derive(Clone, Default)]
struct FailOnDemandAlloc {
    fail: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl FailOnDemandAlloc {
    fn fail_allocations(&self, fail: bool) {
        self.fail.store(fail, std::sync::atomic::Ordering::Relaxed);
    }
}

#[test]
fn write_set_take_transfers_entries_without_cloning() {
    let mut write_set = WriteSet::<TursoAllocator>::new();
    let row_versions: RowVersions<TursoAllocator> = Arc::new(RwLock::new(empty_version_chain()));
    let row_id = RowID::new(MVTableId::from(-2), RowKey::Int(1));

    write_set.push(row_id, Arc::clone(&row_versions));
    let entries_ptr = write_set.entries.as_ptr();
    let entries_capacity = write_set.entries.capacity();
    let strong_count = Arc::strong_count(&row_versions);

    let transferred = write_set.take();

    assert!(write_set.entries.is_empty());
    assert_eq!(transferred.entries.as_ptr(), entries_ptr);
    assert_eq!(transferred.entries.capacity(), entries_capacity);
    assert_eq!(Arc::strong_count(&row_versions), strong_count);
    assert!(Arc::ptr_eq(&transferred.entries[0].1, &row_versions));
}

#[test]
#[should_panic(expected = "write set cannot be modified unless transaction is active")]
fn aborted_transaction_rejects_write_set_insert() {
    let tx = new_tx(1, 1, TransactionState::Aborted);
    let row_versions: RowVersions<TursoAllocator> = Arc::new(RwLock::new(empty_version_chain()));
    let row_id = RowID::new(MVTableId::from(-2), RowKey::Int(1));

    tx.add_to_write_set(row_id, row_versions);
}

#[test]
fn rollback_only_reports_rowids_restored_by_a_delete() {
    let tx_id = 1;
    let row_id = RowID::new(MVTableId::from(-2), RowKey::Int(666));
    let row = Row::new_table_row(row_id, &[]).unwrap();

    let replacement = RowVersion::new(
        1,
        Some(TxTimestampOrID::TxID(tx_id)),
        None,
        row.data.clone(),
        true,
    );
    assert!(!rollback_restores_rowid(tx_id, &replacement));

    let deleted_existing_row = RowVersion::new(
        2,
        Some(TxTimestampOrID::Timestamp(1)),
        Some(TxTimestampOrID::TxID(tx_id)),
        row.data.clone(),
        true,
    );
    assert!(rollback_restores_rowid(tx_id, &deleted_existing_row));

    let deleted_btree_row = RowVersion::new(
        3,
        None,
        Some(TxTimestampOrID::TxID(tx_id)),
        row.data.clone(),
        true,
    );
    assert!(rollback_restores_rowid(tx_id, &deleted_btree_row));

    let deleted_replacement = RowVersion::new(
        4,
        Some(TxTimestampOrID::TxID(tx_id)),
        Some(TxTimestampOrID::TxID(tx_id)),
        row.data.clone(),
        true,
    );
    assert!(rollback_restores_rowid(tx_id, &deleted_replacement));

    let inserted_then_deleted = RowVersion::new(
        5,
        Some(TxTimestampOrID::TxID(tx_id)),
        Some(TxTimestampOrID::TxID(tx_id)),
        row.data,
        false,
    );
    assert!(!rollback_restores_rowid(tx_id, &inserted_then_deleted));
}

unsafe impl crate::alloc::ApiAllocator for FailOnDemandAlloc {
    fn allocate(
        &self,
        layout: crate::alloc::Layout,
    ) -> std::result::Result<std::ptr::NonNull<[u8]>, crate::alloc::AllocError> {
        if self.fail.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(crate::alloc::AllocError);
        }
        <crate::alloc::TursoAllocator as crate::alloc::ApiAllocator>::allocate(
            &crate::alloc::TursoAllocator,
            layout,
        )
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: crate::alloc::Layout) {
        unsafe {
            <crate::alloc::TursoAllocator as crate::alloc::ApiAllocator>::deallocate(
                &crate::alloc::TursoAllocator,
                ptr,
                layout,
            )
        }
    }
}

fn test_mvcc_storage(name: &str) -> Arc<dyn crate::mvcc::persistent_storage::DurableStorage> {
    let io = Arc::new(MemoryIO::new());
    let file = io.open_file(name, OpenFlags::Create, false).unwrap();
    Arc::new(crate::mvcc::persistent_storage::Storage::new(
        file, io, None,
    ))
}

#[test]
fn mv_store_skiplist_allocations_are_fallible() {
    let alloc = FailOnDemandAlloc::default();
    alloc.fail_allocations(true);
    let store = MvStore::new_in(
        MvccClock::new(),
        test_mvcc_storage("mv-store-oom-new.db-log"),
        alloc.clone(),
        false,
    );
    assert!(matches!(store, Err(LimboError::OutOfMemory)));

    alloc.fail_allocations(false);
    let store = MvStore::new_in(
        MvccClock::new(),
        test_mvcc_storage("mv-store-oom-insert.db-log"),
        alloc.clone(),
        false,
    )
    .unwrap();
    alloc.fail_allocations(true);

    let row_id = RowID::new(MVTableId::from(-2), RowKey::Int(1));
    let row_version = RowVersion {
        id: 1,
        begin: PackedTs::pack(Some(TxTimestampOrID::TxID(1))),
        end: PackedTs::pack(None),
        data: Row::new_table_row(row_id.clone(), &[]).unwrap().data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };
    let result = store.insert_version(row_id, row_version);
    assert!(matches!(result, Err(crate::alloc::TryReserveError)));
    assert!(store.rows.is_empty());
}

#[cfg(nightly)]
#[test]
fn row_payload_allocation_uses_passed_allocator() {
    let alloc = FailOnDemandAlloc::default();
    let row_id = RowID::new(MVTableId::from(-2), RowKey::Int(1));

    alloc.fail_allocations(true);
    let result = Row::new_table_row_in(row_id.clone(), &[1, 2, 3], alloc.clone());
    assert!(matches!(result, Err(crate::alloc::TryReserveError)));

    alloc.fail_allocations(false);
    let row = Row::new_table_row_in(row_id, &[1, 2, 3], alloc).unwrap();
    assert_eq!(row.payload(), &[1, 2, 3]);
}

#[cfg(nightly)]
#[test]
fn index_key_payload_allocation_uses_passed_allocator() {
    let alloc = FailOnDemandAlloc::default();
    let record = ImmutableRecord::from_values(&[Value::from_i64(42)], 1).unwrap();
    let record_ref = ImmutableRecordRef::from_bin_record(record.get_payload());
    let index_info = Arc::new(
        IndexInfo::new(
            [crate::types::KeyInfo {
                sort_order: turso_parser::ast::SortOrder::Asc,
                collation: crate::translate::collate::CollationSeq::BINARY,
                nulls_order: None,
            }],
            false,
            1,
            false,
        )
        .unwrap(),
    );

    alloc.fail_allocations(true);
    let result = SortableIndexKey::new_from_payload_in(&record, index_info.clone(), alloc.clone());
    assert!(matches!(result, Err(crate::alloc::TryReserveError)));

    alloc.fail_allocations(false);
    let key = SortableIndexKey::new_from_payload_in(record_ref, index_info, alloc).unwrap();
    assert_eq!(key.key.get_payload(), record.get_payload());
}

#[cfg(nightly)]
#[test]
#[should_panic(expected = "empty tombstone row")]
fn seqcompact_tombstone_payload_uses_store_allocator() {
    let alloc = FailOnDemandAlloc::default();
    let store = MvStore::new_in(
        MvccClock::new(),
        test_mvcc_storage("mv-store-seqcompact-allocator.db-log"),
        alloc.clone(),
        false,
    )
    .unwrap();
    let row_id = RowID::new(MVTableId::from(-2), RowKey::Int(1));
    let versions = store
        .get_or_create_table_row_versions(row_id.clone())
        .unwrap();
    versions.write().try_reserve(1).unwrap();

    alloc.fail_allocations(true);
    store.seqcompact_commit_delete(row_id, 10);
}

#[test]
fn mv_store_insert_allocation_failure_leaves_tx_state_untouched() {
    let alloc = FailOnDemandAlloc::default();
    let store = MvStore::new_in(
        MvccClock::new(),
        test_mvcc_storage("mv-store-oom-insert-ordering.db-log"),
        alloc.clone(),
        false,
    )
    .unwrap();

    let tx_id = 7;
    let tx = new_tx_in::<FailOnDemandAlloc>(tx_id, 1, TransactionState::Active);
    tx.begin_savepoint();
    store.txs.try_insert(tx_id, tx).unwrap();

    let table_id = MVTableId::from(-2);
    let row_id = RowID::new(table_id, RowKey::Int(42));
    let row = Row::new_table_row(row_id.clone(), &[]).unwrap();
    let allocator = store.get_rowid_allocator(&table_id);

    alloc.fail_allocations(true);
    let result = store.insert(tx_id, row);
    assert!(matches!(result, Err(LimboError::OutOfMemory)));

    assert!(store.rows.get(&row_id).is_none());
    assert_eq!(allocator.max_rowid.load(Ordering::SeqCst), 0);

    let tx = store.txs.get(&tx_id).unwrap();
    let tx = tx.value();
    assert!(tx.write_set.lock().is_empty());

    let savepoints = tx.savepoint_stack.read();
    let savepoint = savepoints.last().unwrap();
    assert!(savepoint.created_table_versions.is_empty());
}

#[test]
fn mvcc_passive_gc_retains_until_reader_mark_reaches_materialization() {
    use crate::mvcc::database::WalPos;
    let frame = |f: u64| WalPos {
        checkpoint_seq: 1,
        frame: f,
    };

    // Sole current insert (begin=Ts(5), end=None), materialized at WAL frame 100.
    let stamped_insert = || -> RowVersionChain {
        let mut rv = make_rv(ts(5), None);
        rv.set_materialized_at(frame(100));
        crate::alloc::vec![rv].into()
    };
    // Superseded delete (begin=Ts(3), end=Ts(5)<=lwm), materialized at frame 100.
    let stamped_delete = || -> RowVersionChain {
        let mut rv = make_rv(ts(3), ts(5));
        rv.set_materialized_at(frame(100));
        crate::alloc::vec![rv].into()
    };

    // Reader pinned below the materialization frame: keep both current and delete.
    for mut v in [stamped_insert(), stamped_delete()] {
        let dropped =
            MvStore::<MvccClock>::gc_version_chain(&mut v, 10, 10, true, frame(50), false);
        assert_eq!(
            dropped, 0,
            "version needed by a reader pinned below frame 100 must be kept"
        );
        assert_eq!(v.len(), 1);
    }

    // Readers at the materialization frame: Rule 2 drops deletes; Passive keeps
    // currents unless drop_current_if_in_btree (Truncate).
    let mut deleted = stamped_delete();
    let dropped =
        MvStore::<MvccClock>::gc_version_chain(&mut deleted, 10, 10, true, frame(100), false);
    assert_eq!(
        dropped, 1,
        "materialized + reader-reachable superseded delete must be reclaimed"
    );
    assert!(deleted.is_empty());

    let mut current = stamped_insert();
    let dropped =
        MvStore::<MvccClock>::gc_version_chain(&mut current, 10, 10, true, frame(100), false);
    assert_eq!(
        dropped, 0,
        "Passive keeps the current SkipMap version when drop_current_if_in_btree is false"
    );
    assert_eq!(current.len(), 1);

    let mut current = stamped_insert();
    let dropped =
        MvStore::<MvccClock>::gc_version_chain(&mut current, 10, 10, true, frame(100), true);
    assert_eq!(
        dropped, 0,
        "Passive keeps a live copy while a snapshot is open, even when materialized"
    );
    assert_eq!(current.len(), 1);

    let mut current = stamped_insert();
    let dropped =
        MvStore::<MvccClock>::gc_version_chain(&mut current, u64::MAX, 10, true, frame(100), true);
    assert_eq!(
        dropped, 1,
        "Passive Rule 3 drops a materialized current once no snapshot is open"
    );
    assert!(current.is_empty());

    // Unmaterialized versions are never reclaimed on the Passive path.
    for drop_current_if_in_btree in [false, true] {
        let mut v: RowVersionChain = crate::alloc::vec![make_rv(ts(5), None)].into();
        let dropped = MvStore::<MvccClock>::gc_version_chain(
            &mut v,
            10,
            10,
            true,
            WalPos::STAGED,
            drop_current_if_in_btree,
        );
        assert_eq!(
            dropped, 0,
            "passive GC must not reclaim a version not yet in the B-tree"
        );
        assert_eq!(v.len(), 1);
    }
}

pub(crate) fn generate_simple_string_row(table_id: MVTableId, id: i64, data: &str) -> Row {
    let record =
        ImmutableRecord::from_values(&[Value::Text(Text::new(data.to_string()))], 1).unwrap();
    Row::new_table_row(RowID::new(table_id, RowKey::Int(id)), record.as_blob()).unwrap()
}

// Test for the visibility to check if a new transaction can see old committed values.
// This test checks for the typo present in the paper, explained in https://github.com/penberg/mvcc-rs/issues/15

// Test to check if a older transaction can see (un)committed future rows

use crate::mvcc::database::{MvStore, Row, RowID};
use crate::types::Text;
use crate::MemoryIO;
use crate::Value;
// Simple atomic clock implementation for testing

/* States described in the Hekaton paper *for serializability*:

Table 1: Case analysis of action to take when version V’s
Begin field contains the ID of transaction TB
------------------------------------------------------------------------------------------------------
TB’s state   | TB’s end timestamp | Action to take when transaction T checks visibility of version V.
------------------------------------------------------------------------------------------------------
Active       | Not set            | V is visible only if TB=T and V’s end timestamp equals infinity.
------------------------------------------------------------------------------------------------------
Preparing    | TS                 | V’s begin timestamp will be TS ut V is not yet committed. Use TS
                                  | as V’s begin time when testing visibility. If the test is true,
                                  | allow T to speculatively read V. Committed TS V’s begin timestamp
                                  | will be TS and V is committed. Use TS as V’s begin time to test
                                  | visibility.
------------------------------------------------------------------------------------------------------
Committed    | TS                 | V’s begin timestamp will be TS and V is committed. Use TS as V’s
                                  | begin time to test visibility.
------------------------------------------------------------------------------------------------------
Aborted      | Irrelevant         | Ignore V; it’s a garbage version.
------------------------------------------------------------------------------------------------------
Terminated   | Irrelevant         | Reread V’s Begin field. TB has terminated so it must have finalized
or not found |                    | the timestamp.
------------------------------------------------------------------------------------------------------

Table 2: Case analysis of action to take when V's End field
contains a transaction ID TE.
------------------------------------------------------------------------------------------------------
TE’s state   | TE’s end timestamp | Action to take when transaction T checks visibility of a version V
             |                    | as of read time RT.
------------------------------------------------------------------------------------------------------
Active       | Not set            | V is visible only if TE is not T.
------------------------------------------------------------------------------------------------------
Preparing    | TS                 | V’s end timestamp will be TS provided that TE commits. If TS > RT,
                                  | V is visible to T. If TS < RT, T speculatively ignores V.
------------------------------------------------------------------------------------------------------
Committed    | TS                 | V’s end timestamp will be TS and V is committed. Use TS as V’s end
                                  | timestamp when testing visibility.
------------------------------------------------------------------------------------------------------
Aborted      | Irrelevant         | V is visible.
------------------------------------------------------------------------------------------------------
Terminated   | Irrelevant         | Reread V’s End field. TE has terminated so it must have finalized
or not found |                    | the timestamp.
*/

fn new_tx(tx_id: TxID, begin_ts: u64, state: TransactionState) -> Transaction {
    new_tx_in(tx_id, begin_ts, state)
}

fn new_tx_in<A: super::RowVersionAllocator>(
    tx_id: TxID,
    begin_ts: u64,
    state: TransactionState,
) -> Transaction<A> {
    let state = state.into();
    Transaction {
        state,
        tx_id,
        begin_ts,
        write_set: Mutex::new(WriteSet::new()),
        header: RwLock::new(DatabaseHeader::default()),
        header_dirty: AtomicBool::new(false),
        savepoint_stack: RwLock::new(Vec::new()),
        pager_commit_lock_held: AtomicBool::new(false),
        log_appended: AtomicBool::new(false),
        commit_dep_counter: AtomicU64::new(0),
        abort_now: AtomicBool::new(false),
        commit_dep_set: Mutex::new(HashSet::default()),
        holds_blocking_checkpoint_read: AtomicBool::new(false),
        schema_generation_at_begin: 0,
        read_mark: crate::mvcc::database::WalPos::ORIGIN,
        created_table_ids: Mutex::new(Vec::new()),
        portable_logical_changes: false,
    }
}

/// What this test checks: MVCC transaction visibility and conflict handling follow the intended isolation behavior.
/// Why this matters: Concurrency bugs are correctness bugs: they create anomalies users can observe as wrong query results.
#[test]
fn test_snapshot_isolation_tx_visible1() {
    let txs: SkipMap<TxID, Transaction> = SkipMap::from_iter([
        (1, new_tx(1, 1, TransactionState::Committed(2))),
        (2, new_tx(2, 2, TransactionState::Committed(5))),
        (3, new_tx(3, 3, TransactionState::Aborted)),
        (5, new_tx(5, 5, TransactionState::Preparing(8))),
        (6, new_tx(6, 6, TransactionState::Committed(10))),
        (7, new_tx(7, 7, TransactionState::Active)),
        // tx 8 with Preparing(3): current_tx (begin_ts=4) can speculatively read
        (8, new_tx(8, 1, TransactionState::Preparing(3))),
    ]);
    let finalized_tx_states: SkipMap<TxID, TransactionState> = SkipMap::new();

    let current_tx = new_tx(4, 4, TransactionState::Preparing(7));

    let rv_visible = |begin: Option<TxTimestampOrID>, end: Option<TxTimestampOrID>| {
        let row_version = RowVersion {
            id: 0, // Dummy ID for visibility tests
            begin: crate::mvcc::database::PackedTs::pack(begin),
            end: crate::mvcc::database::PackedTs::pack(end),
            data: generate_simple_string_row((-2).into(), 1, "testme").data,
            btree_resident: false,
            materialized_at: crate::mvcc::database::WalPos::ORIGIN,
        };
        tracing::debug!("Testing visibility of {row_version:?}");
        row_version.is_visible_to(&current_tx, &txs, &finalized_tx_states)
    };

    // begin visible:   transaction committed with ts < current_tx.begin_ts
    // end visible:     inf
    assert!(rv_visible(Some(TxTimestampOrID::TxID(1)), None));

    // begin invisible: transaction committed with ts > current_tx.begin_ts
    assert!(!rv_visible(Some(TxTimestampOrID::TxID(2)), None));

    // begin invisible: transaction aborted
    assert!(!rv_visible(Some(TxTimestampOrID::TxID(3)), None));

    // begin visible:   timestamp < current_tx.begin_ts
    // end invisible:   transaction committed with ts > current_tx.begin_ts
    assert!(!rv_visible(
        Some(TxTimestampOrID::Timestamp(0)),
        Some(TxTimestampOrID::TxID(1))
    ));

    // begin visible:   timestamp < current_tx.begin_ts
    // end visible:     transaction committed with ts < current_tx.begin_ts
    assert!(rv_visible(
        Some(TxTimestampOrID::Timestamp(0)),
        Some(TxTimestampOrID::TxID(2))
    ));

    // begin visible:   timestamp < current_tx.begin_ts
    // end visible:     transaction aborted, delete never happened (Table 2)
    assert!(rv_visible(
        Some(TxTimestampOrID::Timestamp(0)),
        Some(TxTimestampOrID::TxID(3))
    ));

    // begin invisible: transaction preparing with end_ts(8) > begin_ts(4)
    // Speculative read condition (begin_ts >= end_ts) is false: 4 >= 8 is false
    assert!(!rv_visible(Some(TxTimestampOrID::TxID(5)), None));

    // begin VISIBLE via speculative read: tx 8 is Preparing(3), begin_ts(4) >= end_ts(3)
    // Hekaton Table 1: speculatively read and register commit dependency
    assert!(rv_visible(Some(TxTimestampOrID::TxID(8)), None));
    // Verify dependency was registered via register-and-report protocol
    assert_eq!(
        current_tx.commit_dep_counter.load(Ordering::Acquire),
        1,
        "speculative read should register a commit dependency"
    );

    // begin invisible: transaction committed with ts > current_tx.begin_ts
    assert!(!rv_visible(Some(TxTimestampOrID::TxID(6)), None));

    // begin invisible: transaction active
    assert!(!rv_visible(Some(TxTimestampOrID::TxID(7)), None));

    // begin invisible: transaction committed with ts > current_tx.begin_ts
    assert!(!rv_visible(Some(TxTimestampOrID::TxID(6)), None));

    // begin invisible:   transaction active
    assert!(!rv_visible(Some(TxTimestampOrID::TxID(7)), None));

    // begin visible:   timestamp < current_tx.begin_ts
    // end visible:     transaction preparing with TS(8) > RT(4) (Table 2)
    assert!(rv_visible(
        Some(TxTimestampOrID::Timestamp(0)),
        Some(TxTimestampOrID::TxID(5))
    ));

    // begin invisible: timestamp > current_tx.begin_ts
    assert!(!rv_visible(
        Some(TxTimestampOrID::Timestamp(6)),
        Some(TxTimestampOrID::TxID(6))
    ));

    // begin visible:   timestamp < current_tx.begin_ts
    // end visible:     some active transaction will eventually overwrite this version,
    //                  but that hasn't happened
    //                  (this is the https://avi.im/blag/2023/hekaton-paper-typo/ case, I believe!)
    assert!(rv_visible(
        Some(TxTimestampOrID::Timestamp(0)),
        Some(TxTimestampOrID::TxID(7))
    ));

    assert!(!rv_visible(None, None));
}

#[test]
fn test_visibility_uses_finalized_state_for_removed_committed_tx() {
    let txs: SkipMap<TxID, Transaction> = SkipMap::new();
    let finalized_tx_states: SkipMap<TxID, TransactionState> =
        SkipMap::from_iter([(42, TransactionState::Committed(5))]);
    let reader = new_tx(7, 10, TransactionState::Active);

    let inserted_row = RowVersion {
        id: 1,
        begin: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::TxID(42))),
        end: crate::mvcc::database::PackedTs::pack(None),
        data: generate_simple_string_row((-2).into(), 1, "x").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };
    assert!(
        inserted_row.is_visible_to(&reader, &txs, &finalized_tx_states),
        "stale begin=TxID should resolve via finalized committed state"
    );

    let deleted_row = RowVersion {
        id: 2,
        begin: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::Timestamp(1))),
        end: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::TxID(42))),
        data: generate_simple_string_row((-2).into(), 2, "y").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };
    assert!(
        !deleted_row.is_visible_to(&reader, &txs, &finalized_tx_states),
        "stale end=TxID should resolve via finalized committed state"
    );
}

/// Test Hekaton register-and-report: speculative read increments CommitDepCounter
/// and adds to CommitDepSet.
#[test]
fn test_commit_dependency_speculative_read() {
    let txs: SkipMap<TxID, Transaction> =
        SkipMap::from_iter([(1, new_tx(1, 1, TransactionState::Preparing(5)))]);
    let finalized_tx_states: SkipMap<TxID, TransactionState> = SkipMap::new();

    // Reader with begin_ts=10 > end_ts=5 → speculative read → dependency
    let reader = new_tx(2, 10, TransactionState::Active);

    let rv = RowVersion {
        id: 0,
        begin: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::TxID(1))),
        end: crate::mvcc::database::PackedTs::pack(None),
        data: generate_simple_string_row((-2).into(), 1, "test").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };

    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 0);

    // Speculative read: begin_ts(10) >= end_ts(5) → visible, dependency registered
    assert!(rv.is_visible_to(&reader, &txs, &finalized_tx_states));
    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 1);

    // Verify tx 1's CommitDepSet contains reader's tx_id
    let dep_set = txs.get(&1).unwrap();
    assert_eq!(
        *dep_set.value().commit_dep_set.lock(),
        HashSet::from_iter([2])
    );
}

/// Test cascade abort: when depended-on tx aborts, it sets AbortNow on dependents
/// and decrements their CommitDepCounter.
#[test]
fn test_commit_dependency_cascade_abort() {
    let txs: SkipMap<TxID, Transaction> =
        SkipMap::from_iter([(1, new_tx(1, 1, TransactionState::Preparing(5)))]);
    let finalized_tx_states: SkipMap<TxID, TransactionState> = SkipMap::new();

    let reader = new_tx(2, 10, TransactionState::Active);

    let rv = RowVersion {
        id: 0,
        begin: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::TxID(1))),
        end: crate::mvcc::database::PackedTs::pack(None),
        data: generate_simple_string_row((-2).into(), 1, "test").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };

    // Speculative read registers dependency
    assert!(rv.is_visible_to(&reader, &txs, &finalized_tx_states));
    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 1);
    assert!(!reader.abort_now.load(Ordering::Acquire));

    // Simulate tx 1 aborting and cascading to dependents
    let tx1 = txs.get(&1).unwrap();
    let tx1 = tx1.value();
    tx1.state.store(TransactionState::Aborted);

    // Add reader to txs so cascade can find it
    txs.insert(2, reader);

    for dep_tx_id in tx1.commit_dep_set.lock().drain() {
        if let Some(dep_tx_entry) = txs.get(&dep_tx_id) {
            let dep_tx = dep_tx_entry.value();
            dep_tx.abort_now.store(true, Ordering::Release);
            dep_tx.commit_dep_counter.fetch_sub(1, Ordering::AcqRel);
        }
    }

    let reader = txs.get(&2).unwrap();
    let reader = reader.value();
    assert!(reader.abort_now.load(Ordering::Acquire));
    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 0);
}

/// Test that registering a dependency on an already-committed tx is a no-op.
#[test]
fn test_commit_dependency_already_committed() {
    let txs: SkipMap<TxID, Transaction> =
        SkipMap::from_iter([(1, new_tx(1, 1, TransactionState::Committed(5)))]);

    let reader = new_tx(2, 10, TransactionState::Active);

    register_commit_dependency(&txs, &reader, 1);

    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 0);
    assert!(!reader.abort_now.load(Ordering::Acquire));
}

/// Test that registering a dependency on an already-aborted tx sets AbortNow.
#[test]
fn test_commit_dependency_already_aborted() {
    let txs: SkipMap<TxID, Transaction> =
        SkipMap::from_iter([(1, new_tx(1, 1, TransactionState::Aborted))]);

    let reader = new_tx(2, 10, TransactionState::Active);

    register_commit_dependency(&txs, &reader, 1);

    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 0);
    assert!(reader.abort_now.load(Ordering::Acquire));
}

/// Test speculative ignore in is_end_visible registers dependency.
#[test]
fn test_commit_dependency_speculative_ignore() {
    let txs: SkipMap<TxID, Transaction> = SkipMap::from_iter([
        (1, new_tx(1, 1, TransactionState::Committed(2))),
        (3, new_tx(3, 3, TransactionState::Preparing(5))),
    ]);
    let finalized_tx_states: SkipMap<TxID, TransactionState> = SkipMap::new();

    // Reader with begin_ts=10 > end_ts=5: will speculatively ignore (treat as deleted)
    let reader = new_tx(4, 10, TransactionState::Active);

    let rv = RowVersion {
        id: 0,
        begin: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::Timestamp(2))),
        end: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::TxID(3))),
        data: generate_simple_string_row((-2).into(), 1, "test").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };

    // is_end_visible: Preparing(5), begin_ts(10) < 5 = false → deletion visible
    // is_begin_visible: Timestamp(2), 10 >= 2 = true
    // Combined: true && false = false (row not visible because it was deleted)
    assert!(!rv.is_visible_to(&reader, &txs, &finalized_tx_states));
    assert_eq!(
        reader.commit_dep_counter.load(Ordering::Acquire),
        1,
        "speculative ignore should register a commit dependency"
    );
}

/// Test that multiple speculative reads from the same preparing tx only
/// register one commit dependency (dedup).
#[test]
fn test_commit_dependency_multiple_reads_dedup() {
    let txs: SkipMap<TxID, Transaction> =
        SkipMap::from_iter([(1, new_tx(1, 1, TransactionState::Preparing(5)))]);
    let finalized_tx_states: SkipMap<TxID, TransactionState> = SkipMap::new();

    let reader = new_tx(2, 10, TransactionState::Active);

    let make_rv = |row_id: i64| RowVersion {
        id: row_id as u64,
        begin: crate::mvcc::database::PackedTs::pack(Some(TxTimestampOrID::TxID(1))),
        end: crate::mvcc::database::PackedTs::pack(None),
        data: generate_simple_string_row((-2).into(), row_id, "test").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };

    // Read 3 rows from the same preparing tx — dependency is deduplicated
    assert!(make_rv(1).is_visible_to(&reader, &txs, &finalized_tx_states));
    assert!(make_rv(2).is_visible_to(&reader, &txs, &finalized_tx_states));
    assert!(make_rv(3).is_visible_to(&reader, &txs, &finalized_tx_states));

    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 1);

    // tx 1's CommitDepSet has 1 entry for reader (deduplicated)
    let dep_set = txs.get(&1).unwrap();
    assert_eq!(dep_set.value().commit_dep_set.lock().len(), 1);
}

/// Test that register_commit_dependency increments counter before pushing to
/// dep_set, preventing underflow. If counter is incremented after push+unlock,
/// a concurrent drain could fetch_sub(1) on a zero counter, wrapping to MAX.
#[test]
fn test_commit_dependency_counter_no_underflow() {
    let txs: SkipMap<TxID, Transaction> =
        SkipMap::from_iter([(1, new_tx(1, 1, TransactionState::Preparing(5)))]);
    let reader = new_tx(2, 10, TransactionState::Active);

    // Register dependency: counter should go 0 → 1
    register_commit_dependency(&txs, &reader, 1);
    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 1);

    // Simulate drain (as in CommitEnd): fetch_sub should go 1 → 0, not wrap
    reader.commit_dep_counter.fetch_sub(1, Ordering::AcqRel);
    assert_eq!(
        reader.commit_dep_counter.load(Ordering::Acquire),
        0,
        "counter should be exactly 0, not u64::MAX (underflow)"
    );
}

/// Test that registering a dependency on a Terminated (aborted+removed from map)
/// transaction correctly sets AbortNow. Before the fix, rollback_tx removed the
/// tx from txs, so register_commit_dependency saw None and assumed "committed."
#[test]
fn test_commit_dependency_terminated_tx_sets_abort() {
    let txs: SkipMap<TxID, Transaction> =
        SkipMap::from_iter([(1, new_tx(1, 1, TransactionState::Terminated))]);

    let reader = new_tx(2, 10, TransactionState::Active);
    register_commit_dependency(&txs, &reader, 1);

    // Terminated means the tx aborted — must set abort_now
    assert!(
        reader.abort_now.load(Ordering::Acquire),
        "dependency on Terminated tx should set abort_now"
    );
    assert_eq!(
        reader.commit_dep_counter.load(Ordering::Acquire),
        0,
        "no counter increment for aborted/terminated dependency"
    );
}

/// Test that when tx is NOT in the map (removed), register_commit_dependency
/// treats it as committed (no abort_now, no counter increment). This is correct
/// only for committed transactions. Aborted transactions should NOT be removed
/// from the map (Issue #3 fix ensures this).
#[test]
fn test_commit_dependency_missing_tx_assumes_committed() {
    let txs: SkipMap<TxID, Transaction> = SkipMap::new();

    let reader = new_tx(2, 10, TransactionState::Active);
    register_commit_dependency(&txs, &reader, 99);

    assert!(
        !reader.abort_now.load(Ordering::Acquire),
        "missing tx (committed+removed) should not set abort_now"
    );
    assert_eq!(reader.commit_dep_counter.load(Ordering::Acquire), 0);
}

/// The implementation maintains the intended invariant for this scenario.
#[test]
fn transaction_display() {
    let state = AtomicTransactionState::from(TransactionState::Preparing(20250915));
    let tx_id = 42;
    let begin_ts = 20250914;

    let empty_versions = || Arc::new(RwLock::new(empty_version_chain()));
    let write_set = Mutex::new({
        let mut write_set: WriteSet = WriteSet::new();
        write_set.push(RowID::new((-2).into(), RowKey::Int(11)), empty_versions());
        write_set.push(RowID::new((-2).into(), RowKey::Int(13)), empty_versions());
        write_set
    });

    let tx = Transaction {
        state,
        tx_id,
        begin_ts,
        write_set,
        header: RwLock::new(DatabaseHeader::default()),
        header_dirty: AtomicBool::new(false),
        savepoint_stack: RwLock::new(Vec::new()),
        pager_commit_lock_held: AtomicBool::new(false),
        log_appended: AtomicBool::new(false),
        commit_dep_counter: AtomicU64::new(0),
        abort_now: AtomicBool::new(false),
        commit_dep_set: Mutex::new(HashSet::default()),
        holds_blocking_checkpoint_read: AtomicBool::new(false),
        schema_generation_at_begin: 0,
        read_mark: crate::mvcc::database::WalPos::ORIGIN,
        created_table_ids: Mutex::new(Vec::new()),
        portable_logical_changes: false,
    };

    let expected = "{ state: Preparing(20250915), id: 42, begin_ts: 20250914, write_set: [RowID { table_id: MVTableId(-2), row_id: Int(11) }, RowID { table_id: MVTableId(-2), row_id: Int(13) }] }";
    let output = format!("{tx}");
    assert_eq!(output, expected);
}

// ─── GC helpers ───────────────────────────────────────────────────────────

fn make_rv(begin: Option<TxTimestampOrID>, end: Option<TxTimestampOrID>) -> RowVersion {
    RowVersion {
        id: 0,
        begin: crate::mvcc::database::PackedTs::pack(begin),
        end: crate::mvcc::database::PackedTs::pack(end),
        data: generate_simple_string_row((-2).into(), 1, "gc_test").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    }
}

/// [`make_rv`] for a version a checkpoint already wrote to the B-tree, which is
/// what Rule 3 requires before it may drop the last SkipMap copy.
fn make_materialized_rv(
    begin: Option<TxTimestampOrID>,
    end: Option<TxTimestampOrID>,
) -> RowVersion {
    let mut rv = make_rv(begin, end);
    rv.set_materialized_at(crate::mvcc::database::WalPos {
        checkpoint_seq: 1,
        frame: 1,
    });
    rv
}

fn ts(v: u64) -> Option<TxTimestampOrID> {
    Some(TxTimestampOrID::Timestamp(v))
}

fn txid(v: u64) -> Option<TxTimestampOrID> {
    Some(TxTimestampOrID::TxID(v))
}

// ─── GC unit tests ───────────────────────────────────────────────────────

#[test]
/// Rolled-back transactions leave versions with begin=None, end=None. These are
/// invisible to every transaction and must be removed unconditionally by Rule 1.
fn test_gc_rule1_aborted_garbage_removed() {
    let mut versions: RowVersionChain = crate::alloc::vec![make_rv(None, None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1);
    assert!(versions.is_empty());
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Rule 1 removes only aborted garbage, leaving live and superseded versions intact.
fn test_gc_rule1_aborted_among_live_versions() {
    let mut versions: RowVersionChain = crate::alloc::vec![
        make_rv(ts(5), None),  // current
        make_rv(None, None),   // aborted
        make_rv(ts(3), ts(5)), // superseded
    ]
    .into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        2,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    // Only aborted removed; superseded has e=5 > lwm=2 so retained
    assert_eq!(dropped, 1);
    assert_eq!(versions.len(), 2);
    assert!(versions
        .iter()
        .all(|rv| rv.begin().is_some() || rv.end().is_some()));
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// A superseded version whose end timestamp is at or below the low-water mark is
/// invisible to all active readers. When a committed current version exists to
/// take over B-tree invalidation, the superseded version is safely removable.
fn test_gc_rule2_superseded_below_lwm_with_current() {
    // Superseded version (end=Timestamp(3)) below LWM=10, and there's a current version.
    let mut versions: RowVersionChain = crate::alloc::vec![
        make_rv(ts(3), ts(5)), // superseded, e=5 <= lwm=10
        make_rv(ts(5), None),  // current
    ]
    .into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1);
    assert_eq!(versions.len(), 1);
    assert!(versions[0].end().is_none()); // only current remains
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// A superseded version whose end timestamp exceeds the LWM may still be visible
/// to an active reader. It must be retained regardless of other conditions.
fn test_gc_rule2_superseded_above_lwm_retained() {
    // Superseded version (end=Timestamp(15)) above LWM=10 — must be retained.
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_rv(ts(3), ts(15)), make_rv(ts(15), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 2);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// When a row was deleted but the deletion hasn't been checkpointed to the B-tree
/// yet (e > ckpt_max), the tombstone is the only thing hiding the stale B-tree
/// row. Removing it would resurrect a deleted row. Must be retained.
fn test_gc_rule2_tombstone_guard_uncheckpointed() {
    // Tombstone: end is set, no current version, and e > ckpt_max.
    // Must be retained to prevent row resurrection via dual cursor.
    let mut versions: RowVersionChain = crate::alloc::vec![
        make_rv(ts(3), ts(5)), // tombstone (sole version, no current)
    ]
    .into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        2,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    // e=5 > ckpt_max=2, no current → tombstone guard retains it
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 1);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Once the deletion has been checkpointed (e <= ckpt_max), the B-tree no longer
/// contains the row, so the tombstone is safe to remove.
fn test_gc_rule2_tombstone_guard_checkpointed() {
    // Tombstone with e <= ckpt_max — deletion is checkpointed, safe to remove.
    let mut versions: RowVersionChain = crate::alloc::vec![make_rv(ts(3), ts(5))].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    // e=5 <= ckpt_max=5, e=5 <= lwm=10 → removable
    assert_eq!(dropped, 1);
    assert!(versions.is_empty());
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// A current version that's been checkpointed to B-tree, with no other versions in
/// the chain and no transaction open at all, is redundant. The dual cursor will
/// fall through to the B-tree which has identical data. Safe to remove.
fn test_gc_rule3_drop_current_when_in_btree() {
    // Single stamped current with b <= ckpt_max, and no transaction open
    // (lwm == u64::MAX).
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_materialized_rv(ts(5), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1);
    assert!(versions.is_empty());
}

/// Truncate Rule 3 matches Passive: sole stamped currents clear only when idle.
#[test]
fn test_gc_rule3_truncate_idle_only() {
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_materialized_rv(ts(5), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0, "open snapshot must block Truncate Rule 3 clear");
    assert_eq!(versions.len(), 1);

    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1);
    assert!(versions.is_empty());
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// A current version not yet checkpointed (b > ckpt_max) cannot be removed —
/// the B-tree doesn't have the data, so fallthrough would return stale results.
fn test_gc_rule3_not_checkpointed_retained() {
    // Single current version with b > ckpt_max — B-tree doesn't have it yet.
    // Nothing is open (lwm == u64::MAX), so ckpt_max is the only thing holding
    // this version back.
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_materialized_rv(ts(5), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        3,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 1);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Any finite LWM means some transaction is still open, and an open snapshot may
/// be running a dual-cursor scan that would lose this row if the chain emptied
/// under it. Rule 3 only fires when nothing is open, so this is retained.
fn test_gc_rule3_visible_to_active_tx_retained() {
    // Single stamped current, already inside ckpt_max, but a transaction is open.
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_materialized_rv(ts(5), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        5,
        10,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    // lwm=5 != u64::MAX, so a transaction is open and Rule 3 must not fire.
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 1);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// A current version cannot be removed before checkpoint has persisted it.
fn test_gc_rule3_current_retained_before_first_checkpoint() {
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_materialized_rv(ts(1), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 1);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Once checkpoint has persisted a sole current version, it becomes GC-eligible.
fn test_gc_rule3_current_collected_after_checkpoint() {
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_materialized_rv(ts(1), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1);
    assert_eq!(versions.len(), 0);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Rule 3 only drops the last remaining current version. After Rule 2 removes
/// superseded history, Rule 3 can then drop that current.
fn test_gc_rule3_after_history_reclaimed() {
    // Rule 3 only fires when exactly one version remains after rules 1 & 2.
    let mut versions: RowVersionChain =
        crate::alloc::vec![make_rv(ts(3), ts(5)), make_materialized_rv(ts(5), None)].into();
    // Both versions are inside ckpt_max and nothing is open, but the chain
    // starts with 2 entries. Rule 2 removes the superseded one
    // (has_current=true), then Rule 3 drops the remaining current.
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 2);
    assert!(versions.is_empty());
}

/// Rule 3 drops a sole current only once a checkpoint stamped it. A high
/// `ckpt_max` says a checkpoint ran, not that this row reached a B-tree leaf.
#[test]
fn test_gc_rule3_keeps_unstamped_current_and_drops_stamped_one() {
    let mut versions: RowVersionChain = crate::alloc::vec![make_rv(ts(5), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        10,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 1);

    versions[0].set_materialized_at(crate::mvcc::database::WalPos {
        checkpoint_seq: 1,
        frame: 7,
    });
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        10,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1);
    assert!(versions.is_empty());
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Versions referencing an active transaction (begin=TxID) represent uncommitted
/// inserts. They don't match any removal rule and must always be retained.
fn test_gc_txid_refs_retained() {
    // Versions with TxID (uncommitted) references are never collected.
    let mut versions: RowVersionChain = crate::alloc::vec![make_rv(txid(99), None)].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        u64::MAX,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 1);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Versions with end=TxID represent an uncommitted deletion. Rule 2 only matches
/// end=Timestamp, so these are never collected until the deleting tx resolves.
fn test_gc_txid_end_retained() {
    // end=TxID means the deletion is uncommitted; rule 2 only matches Timestamp.
    let mut versions: RowVersionChain = crate::alloc::vec![make_rv(ts(3), txid(50))].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        u64::MAX,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 1);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// A pending insert (begin=TxID) must NOT count as a "committed current version"
/// for the tombstone guard. If it rolled back, the tombstone would be the only
/// thing hiding the stale B-tree row, and removing it would resurrect deleted data.
fn test_gc_rule2_pending_insert_does_not_disable_tombstone_guard() {
    // A pending insert (begin=TxID, end=None) coexists with a tombstone.
    // has_current must NOT count the pending insert — if it rolls back,
    // the tombstone is the only thing hiding the B-tree row.
    let mut versions: RowVersionChain = crate::alloc::vec![
        make_rv(ts(3), ts(5)), // tombstone: deletion at e=5, not checkpointed (ckpt_max=2)
        make_rv(txid(99), None), // pending insert (uncommitted)
    ]
    .into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        2,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    // Tombstone must be retained: e=5 > ckpt_max=2, and pending insert doesn't count.
    // Only nothing changes (pending insert is not aborted garbage either).
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 2);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// When a committed current version exists (begin=Timestamp, end=None), it takes
/// over MVCC visibility from a non-B-tree superseded version. The tombstone guard
/// is no longer needed, so the superseded version can be safely removed.
fn test_gc_rule2_committed_current_disables_non_btree_tombstone_guard() {
    // A committed current version (begin=Timestamp, end=None) means the row
    // has a live successor — the tombstone can safely be removed.
    let mut versions: RowVersionChain = crate::alloc::vec![
        make_rv(ts(3), ts(5)), // superseded, e=5 <= lwm=10
        make_rv(ts(5), None),  // committed current
    ]
    .into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        2,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    // Superseded removed (has_current=true for committed version), current remains.
    assert_eq!(dropped, 1);
    assert_eq!(versions.len(), 1);
    assert!(versions[0].end().is_none());
}

/// A B-tree-resident version whose ending timestamp has not been checkpointed
/// still records a required physical B-tree delete or overwrite. A committed
/// replacement can hide it from readers but cannot make it GC-eligible until
/// checkpoint makes that physical change durable.
#[test]
fn test_gc_rule2_btree_resident_marker_with_current_retained_until_checkpoint() {
    let mut tombstone = make_rv(None, ts(5));
    tombstone.btree_resident = true;
    let current = make_materialized_rv(ts(5), None);
    let mut versions: RowVersionChain = crate::alloc::vec![tombstone, current.clone()].into();

    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        2,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 2);
    assert!(versions[0].btree_resident);

    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 2);
    assert!(versions.is_empty());

    let mut rewritten_btree_row = make_rv(ts(3), ts(5));
    rewritten_btree_row.btree_resident = true;
    let mut versions: RowVersionChain = crate::alloc::vec![rewritten_btree_row, current].into();

    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        2,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 2);
    assert!(versions[0].btree_resident);

    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 2);
    assert!(versions.is_empty());
}

/// A superseded version whose insert was checkpointed (begin <= ckpt_max) is
/// physically in the B-tree even when its `btree_resident` flag is unset (the
/// flag is only seeded by the dual cursor, not by checkpoint). Until its
/// overwrite/delete is checkpointed (end > ckpt_max), it must be retained even
/// with a committed current replacement: the checkpointer derives DB-file
/// existence from begin/end timestamps against the durable boundary, so
/// removing this version would make a later delete skip the B-tree write and
/// resurrect the row (issue #7638).
#[test]
fn test_gc_rule2_checkpointed_insert_with_current_retained_until_checkpoint() {
    // begin=2 <= ckpt_max=2 (insert checkpointed), end=5 > ckpt_max (overwrite not).
    let checkpointed_btree_row = make_rv(ts(2), ts(5));
    let current = make_materialized_rv(ts(5), None);
    let mut versions: RowVersionChain = crate::alloc::vec![checkpointed_btree_row, current].into();

    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        2,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 2);

    // Once the overwrite is checkpointed (end <= ckpt_max), both versions are
    // reclaimable (rule 2 for the superseded, rule 3 for the current).
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 2);
    assert!(versions.is_empty());
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// B-tree tombstones (begin=None, end=e) represent rows that existed in the B-tree
/// before MVCC was enabled and were then deleted. Before checkpoint writes the
/// deletion, the tombstone hides the stale B-tree row. After checkpoint, it's safe
/// to remove. Tests the full lifecycle: retained → checkpointed → collected.
fn test_gc_rule2_btree_tombstone_lifecycle() {
    // B-tree tombstone: begin=None, end=Timestamp(e) where e > 0.
    // Represents a row deleted in MVCC that existed in B-tree before MVCC.
    // Before checkpoint (ckpt_max < e): tombstone must be retained.
    let mut versions: RowVersionChain = crate::alloc::vec![make_rv(None, ts(5))].into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        3,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0, "tombstone retained: e=5 > ckpt_max=3");
    assert_eq!(versions.len(), 1);

    // After checkpoint (ckpt_max >= e): tombstone is collected.
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1, "tombstone collected: e=5 <= ckpt_max=5");
    assert_eq!(versions.len(), 0);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// Rule 3 must never fire when superseded versions remain in the chain — removing
/// the current version would leave orphaned superseded versions that "poison" the
/// dual cursor, making it hide the B-tree row without providing a replacement.
fn test_gc_rule3_not_firing_with_unremovable_superseded() {
    // Two versions: superseded with e > lwm (can't remove), and current.
    // Rule 2 can't remove the superseded one, so 2 versions remain.
    // Rule 3 needs a single remaining current, so it must NOT fire.
    let mut versions: RowVersionChain = crate::alloc::vec![
        make_rv(ts(3), ts(15)),             // e=15 > lwm=10 — retained
        make_materialized_rv(ts(15), None), // current
    ]
    .into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        20,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
    assert_eq!(versions.len(), 2);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// GC on an empty version chain is a no-op. Verifies no panics or off-by-one errors.
fn test_gc_noop_on_empty() {
    let mut versions = empty_version_chain();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        10,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 0);
}

/// Garbage collection removes only versions that are provably unreachable and keeps versions still required for visibility and safety.
#[test]
/// All three rules fire together with nothing open: aborted garbage (Rule 1), two
/// checkpointed superseded versions with a committed current (Rule 2), and the sole
/// surviving stamped current inside `ckpt_max` (Rule 3). The chain is fully reclaimed.
fn test_gc_combined_rules() {
    let mut versions: RowVersionChain = crate::alloc::vec![
        make_rv(None, None),   // aborted → rule 1
        make_rv(ts(1), ts(3)), // superseded, e=3 <= ckpt_max=5 → rule 2 (has_current=true)
        make_rv(ts(3), ts(5)), // superseded, e=5 <= ckpt_max=5 → rule 2
        // current, stamped, b=5 <= ckpt_max=5, nothing open → rule 3
        make_materialized_rv(ts(5), None),
    ]
    .into();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        u64::MAX,
        5,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 4);
    assert!(versions.is_empty());
}

/// GC trims chains with retain()/clear(), which keeps the Vec's allocation.
/// After a burst of versions is collected, the chain's capacity must be
/// released down to a quarter of its previous value (deliberately not to fit)
/// so a hot row doesn't pin its peak allocation forever.
#[test]
fn test_gc_shrinks_version_chain_capacity() {
    let make_version = |begin, end| RowVersion {
        id: 0,
        begin: crate::mvcc::database::PackedTs::pack(begin),
        end: crate::mvcc::database::PackedTs::pack(end),
        data: generate_simple_string_row((-2).into(), 1, "shrink").data,
        btree_resident: false,
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    };

    // One committed current version that survives GC (b=1 > ckpt_max=0, so
    // rule 3 doesn't fire), plus a burst of aborted garbage (always removed).
    let mut versions: RowVersionChain<TursoAllocator> =
        std::iter::once(make_version(Some(TxTimestampOrID::Timestamp(1)), None))
            .chain((0..1023).map(|_| make_version(None, None)))
            .try_collect()
            .unwrap();
    let capacity_before = versions.capacity();
    assert!(capacity_before >= 1024);

    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        0,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1023);
    assert_eq!(versions.len(), 1);
    assert!(
        versions.capacity() <= capacity_before / 4,
        "chain capacity should shrink to a quarter of {capacity_before}, got {}",
        versions.capacity()
    );

    // A chain emptied entirely also releases its allocation.
    let mut versions: RowVersionChain<TursoAllocator> = (0..1024)
        .map(|_| make_version(None, None))
        .try_collect()
        .unwrap();
    let capacity_before = versions.capacity();
    let dropped = MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        0,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert_eq!(dropped, 1024);
    assert!(versions.is_empty());
    assert!(
        versions.capacity() <= capacity_before / 4,
        "empty chain capacity should shrink to a quarter of {capacity_before}, got {}",
        versions.capacity()
    );

    // Small chains are not worth a realloc: capacity at or below the minimum
    // threshold is left untouched even when fully emptied.
    let mut small: RowVersionChain<TursoAllocator> =
        <RowVersionChain<TursoAllocator> as TursoTryWithCapacityExt>::try_with_capacity_ext(16)
            .unwrap();
    small.push(make_version(None, None));
    let capacity_before = small.capacity();
    MvStore::<MvccClock>::gc_version_chain(
        &mut small,
        0,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    assert!(small.is_empty());
    // A `Vec` keeps a small allocation. The stable build keeps one version
    // inline, so its chain frees the buffer when one version or none remains.
    #[cfg(nightly)]
    assert_eq!(small.capacity(), capacity_before);
    #[cfg(not(nightly))]
    {
        assert_eq!(capacity_before, 16);
        assert_eq!(small.capacity(), 1);
    }
}

/// A version chain stores its versions inline, so the size of `RowVersion` is a
/// large part of the heap that MVCC uses for each row. The nightly build stores
/// an allocator in each payload, so its sizes are different.
#[cfg(not(nightly))]
#[test]
fn test_row_version_size() {
    assert_eq!(std::mem::size_of::<WalPos>(), 12);
    assert_eq!(std::mem::size_of::<RowVersion>(), 56);
}

/// A version does not store its column count. It reads the count from its
/// record, and a tombstone with an empty payload has no columns.
#[test]
fn test_row_version_column_count_comes_from_the_record() {
    let id = RowID::new(MVTableId::from(-2), RowKey::Int(1));
    let record =
        ImmutableRecord::from_values(&[Value::from_i64(7), Value::build_text("seven")], 2).unwrap();
    let row = Row::new_table_row(id.clone(), record.as_blob()).unwrap();
    let version = RowVersion::new(
        1,
        Some(TxTimestampOrID::Timestamp(1)),
        None,
        row.data,
        false,
    );
    assert_eq!(version.column_count(&id), 2);
    let empty = Row::new_table_row(id.clone(), &[]).unwrap();
    let tombstone = RowVersion::new(
        2,
        None,
        Some(TxTimestampOrID::Timestamp(2)),
        empty.data,
        true,
    );
    assert_eq!(tombstone.column_count(&id), 0);
}

// ─── GC quickcheck property tests ────────────────────────────────────────

/// Represents a version chain entry for quickcheck.
#[derive(Debug, Clone)]
struct ArbitraryVersionChain {
    versions: RowVersionChain<TursoAllocator>,
    lwm: u64,
    ckpt_max: u64,
}

/// Generates RowVersions matching realistic MVCC states.
/// Only produces valid (begin, end) combinations that can actually occur.
fn arbitrary_row_version(g: &mut Gen) -> RowVersion {
    // Weight toward realistic states:
    // 32% current (Timestamp, None), 24% superseded (Timestamp, Timestamp),
    // 8% aborted (None, None), 8% pending insert (TxID, None),
    // 8% pending delete (Timestamp, TxID), 20% B-tree tombstone (None, Timestamp)
    let kind = u8::arbitrary(g) % 25;
    let (begin, end) = match kind {
        0..=7 => {
            // Current committed version
            let b = u64::arbitrary(g) % 20 + 1;
            (Some(TxTimestampOrID::Timestamp(b)), None)
        }
        8..=13 => {
            // Superseded version
            let b = u64::arbitrary(g) % 15 + 1;
            let e = b + u64::arbitrary(g) % 10 + 1;
            (
                Some(TxTimestampOrID::Timestamp(b)),
                Some(TxTimestampOrID::Timestamp(e)),
            )
        }
        14..=15 => {
            // Aborted garbage
            (None, None)
        }
        16..=17 => {
            // Pending insert
            let t = u64::arbitrary(g) % 20 + 1;
            (Some(TxTimestampOrID::TxID(t)), None)
        }
        18..=19 => {
            // Pending delete
            let b = u64::arbitrary(g) % 15 + 1;
            let t = u64::arbitrary(g) % 20 + 1;
            (
                Some(TxTimestampOrID::Timestamp(b)),
                Some(TxTimestampOrID::TxID(t)),
            )
        }
        20..=24 => {
            // B-tree tombstone (begin=None, end=e) — row existed before MVCC, then deleted
            let e = u64::arbitrary(g) % 20 + 1;
            (None, Some(TxTimestampOrID::Timestamp(e)))
        }
        _ => unreachable!(),
    };

    RowVersion {
        id: 0,
        begin: crate::mvcc::database::PackedTs::pack(begin),
        end: crate::mvcc::database::PackedTs::pack(end),
        data: generate_simple_string_row((-2).into(), 1, "qc").data,
        btree_resident: bool::arbitrary(g),
        materialized_at: crate::mvcc::database::WalPos::ORIGIN,
    }
}

impl Arbitrary for ArbitraryVersionChain {
    fn arbitrary(g: &mut Gen) -> Self {
        // 1..8 versions (no empty chains — they trivially pass all properties)
        let len = usize::arbitrary(g) % 8 + 1;
        let versions: RowVersionChain<TursoAllocator> = (0..len)
            .map(|_| arbitrary_row_version(g))
            .try_collect()
            .unwrap();
        // Include boundary values with ~20% probability each.
        let lwm = match u8::arbitrary(g) % 5 {
            0 => 0,
            1 => u64::MAX, // blocking checkpoint case
            _ => u64::arbitrary(g) % 30,
        };
        let ckpt_max = match u8::arbitrary(g) % 5 {
            0 => 0,        // no checkpoint has run
            1 => u64::MAX, // everything checkpointed
            _ => u64::arbitrary(g) % 30,
        };
        Self {
            versions,
            lwm,
            ckpt_max,
        }
    }
}

/// GC only removes versions — it never synthesizes new ones. For any input chain,
/// the output must be a subset (same length or shorter).
#[quickcheck]
fn prop_gc_never_increases_version_count(chain: ArbitraryVersionChain) -> bool {
    let before = chain.versions.len();
    let mut versions = chain.versions;
    MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    versions.len() <= before
}

/// Running GC twice with the same LWM and ckpt_max must produce the same result
/// as running it once. A non-idempotent GC would indicate that GC output triggers
/// further removals on re-evaluation, which means the first pass missed something.
/// Compares actual version content (begin/end), not just chain length.
#[quickcheck]
fn prop_gc_is_idempotent(chain: ArbitraryVersionChain) -> bool {
    let mut v1 = chain.versions.clone();
    MvStore::<MvccClock>::gc_version_chain(
        &mut v1,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    let snapshot = v1.clone();
    MvStore::<MvccClock>::gc_version_chain(
        &mut v1,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    // Compare content, not just length — a swap bug would pass a length-only check.
    v1.len() == snapshot.len()
        && v1
            .iter()
            .zip(snapshot.iter())
            .all(|(a, b)| a.begin() == b.begin() && a.end() == b.end())
}

/// Aborted garbage (begin=None, end=None) is invisible to every transaction and
/// has no B-tree implications. GC must remove all of it unconditionally (Rule 1).
/// No aborted garbage should survive a GC pass, regardless of LWM or ckpt_max.
#[quickcheck]
fn prop_gc_removes_all_aborted_garbage(chain: ArbitraryVersionChain) -> bool {
    let mut versions = chain.versions;
    MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    versions
        .iter()
        .all(|rv| !matches!((&rv.begin(), &rv.end()), (None, None)))
}

/// Uncommitted inserts (begin=TxID, end=None) belong to an in-flight transaction.
/// GC cannot know whether it will commit or abort, so it must never touch them.
/// Verifies all such versions survive GC regardless of other chain contents.
#[quickcheck]
fn prop_gc_retains_txid_begins(chain: ArbitraryVersionChain) -> bool {
    let txid_begins_before: usize = chain
        .versions
        .iter()
        .filter(|rv| matches!(&rv.begin(), Some(TxTimestampOrID::TxID(_))) && rv.end().is_none())
        .count();
    let mut versions = chain.versions;
    MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    let txid_begins_after: usize = versions
        .iter()
        .filter(|rv| matches!(&rv.begin(), Some(TxTimestampOrID::TxID(_))) && rv.end().is_none())
        .count();
    // Active uncommitted versions (begin=TxID, end=None) are never aborted garbage
    // and don't match rule 2 or 3, so they should be retained.
    txid_begins_after == txid_begins_before
}

/// Uncommitted deletions (end=TxID) represent a pending delete by an in-flight
/// transaction. Rule 2 only matches end=Timestamp, so these must be retained.
/// Verifies GC never removes versions with TxID end markers.
#[quickcheck]
fn prop_gc_retains_txid_ends(chain: ArbitraryVersionChain) -> bool {
    // Versions with end=TxID and non-None begin are not matched by any removal
    // rule (rule 1 requires (None,None), rule 2 requires end=Timestamp).
    let filter = |rv: &&RowVersion| {
        matches!(&rv.end(), Some(TxTimestampOrID::TxID(_))) && rv.begin().is_some()
    };
    let txid_ends_before: usize = chain.versions.iter().filter(filter).count();
    let mut versions = chain.versions;
    MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    let txid_ends_after: usize = versions.iter().filter(filter).count();
    txid_ends_after == txid_ends_before
}

/// Current versions (begin=Timestamp(b), end=None) are not removable before they
/// are checkpointed. Forces ckpt_max=0 and verifies all committed current versions
/// survive.
#[quickcheck]
fn prop_gc_current_versions_protected_before_checkpoint(chain: ArbitraryVersionChain) -> bool {
    let current_before: usize = chain
        .versions
        .iter()
        .filter(|rv| {
            matches!(
                (&rv.begin(), &rv.end()),
                (Some(TxTimestampOrID::Timestamp(_)), None)
            )
        })
        .count();
    let mut versions = chain.versions;
    MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        chain.lwm,
        0,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );
    let current_after: usize = versions
        .iter()
        .filter(|rv| {
            matches!(
                (&rv.begin(), &rv.end()),
                (Some(TxTimestampOrID::Timestamp(_)), None)
            )
        })
        .count();
    current_after == current_before
}

/// When a row has been deleted but the deletion isn't checkpointed yet, the
/// tombstone (superseded version with end > ckpt_max) is the only thing preventing
/// the dual cursor from reading a stale B-tree row. If GC empties such a chain,
/// the deleted row reappears. Verifies GC never empties a chain that has
/// uncheckpointed tombstones and no committed current version to take over.
#[quickcheck]
fn prop_gc_tombstone_guard_preserves_btree_safety(chain: ArbitraryVersionChain) -> bool {
    // If a chain has only superseded versions (no committed current) and at
    // least one has e > ckpt_max, GC must not empty the chain — removing all
    // versions would let the dual cursor fall through to a stale B-tree row.
    let mut versions = chain.versions.clone();
    MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );

    // Check: if pre-GC chain had no committed current version AND had a
    // superseded version with e > ckpt_max, post-GC chain must not be empty.
    let had_committed_current = chain
        .versions
        .iter()
        .any(|rv| rv.end().is_none() && matches!(&rv.begin(), Some(TxTimestampOrID::Timestamp(_))));
    let had_uncheckpointed_tombstone = chain
        .versions
        .iter()
        .any(|rv| matches!(&rv.end(), Some(TxTimestampOrID::Timestamp(e)) if *e > chain.ckpt_max));
    // Only non-garbage versions matter (aborted garbage is always removed first)
    let had_non_garbage = chain
        .versions
        .iter()
        .any(|rv| !matches!((&rv.begin(), &rv.end()), (None, None)));

    if !had_committed_current && had_uncheckpointed_tombstone && had_non_garbage {
        !versions.is_empty()
    } else {
        true // no constraint in this case
    }
}

/// Superseded versions without a committed current version are dangerous — their
/// presence tells the dual cursor "this row was modified" but there's no current
/// version to serve reads. GC must only leave such orphans when they're justifiably
/// retained: still visible to a reader (e > lwm), guarding an uncheckpointed
/// deletion (e > ckpt_max).
#[quickcheck]
fn prop_gc_no_orphaned_superseded_versions(chain: ArbitraryVersionChain) -> bool {
    // After GC, if a chain has superseded versions without a committed current
    // version, each superseded version must be justifiably retained:
    // - e > lwm (Rule 2 didn't fire — still visible to some reader)
    // - e > ckpt_max (tombstone guard — deletion not yet in B-tree)
    let mut versions = chain.versions;
    MvStore::<MvccClock>::gc_version_chain(
        &mut versions,
        chain.lwm,
        chain.ckpt_max,
        false,
        crate::mvcc::database::WalPos::STAGED,
        true,
    );

    let has_committed_current = versions
        .iter()
        .any(|rv| rv.end().is_none() && matches!(&rv.begin(), Some(TxTimestampOrID::Timestamp(_))));
    let has_superseded = versions.iter().any(|rv| {
        matches!(
            (&rv.begin(), &rv.end()),
            (
                Some(TxTimestampOrID::Timestamp(_)),
                Some(TxTimestampOrID::Timestamp(_))
            )
        )
    });

    if has_superseded && !has_committed_current {
        versions
            .iter()
            .filter(|rv| {
                matches!(
                    (&rv.begin(), &rv.end()),
                    (
                        Some(TxTimestampOrID::Timestamp(_)),
                        Some(TxTimestampOrID::Timestamp(_))
                    )
                )
            })
            .all(|rv| {
                if let Some(TxTimestampOrID::Timestamp(e)) = &rv.end() {
                    *e > chain.lwm || *e > chain.ckpt_max
                } else {
                    false
                }
            })
    } else {
        true
    }
}

// ─── End-to-end GC + dual cursor tests ───────────────────────────────────

#[cfg(feature = "conn_raw_api")]
#[test]
fn test_mvcc_portable_changes_encoder_matches_metadata_wire_golden() {
    let mut builder = PortableLogicalBuilder::new();
    builder.add_metadata("client", "client-a").unwrap();
    builder
        .add_object_map(PortableObjectMapEntry {
            mv_table_id: -5,
            name: "items",
        })
        .unwrap();
    let encoded = builder.finish().unwrap();

    assert_eq!(
        encoded,
        vec![
            0x62, 0x06, b'c', b'l', b'i', b'e', b'n', b't', 0x62, 0x08, b'c', b'l', b'i', b'e',
            b'n', b't', b'-', b'a', 0x62, 0x05, b'i', b't', b'e', b'm', b's', 0x6a, 0x04, 0x08,
            0x09, // mv_table_id = -5
            0x10, 0x02, // name_ref = "items"
            0x72, 0x04, 0x08, 0x00, // metadata key_ref = "client"
            0x10, 0x01, // metadata value_ref = "client-a"
        ]
    );
}

// https://github.com/tursodatabase/turso/issues/6752

// https://github.com/tursodatabase/turso/issues/6757

// https://github.com/tursodatabase/turso/issues/6755

fn empty_version_chain() -> RowVersionChain<TursoAllocator> {
    <RowVersionChain<TursoAllocator> as TursoVecInExt<RowVersion, TursoAllocator>>::new_in(
        TursoAllocator,
    )
}
