use super::*;

const OWNED_STATE: Inheritance = Inheritance {
    pipeline: StateSource::Command,
    buffers: StateSource::Command,
};

const TEST_DESCRIPTOR: CommandDescriptor = CommandDescriptor::Compute {
    dispatch: ComputeDispatch::Both,
    max_buffer_bind_count: 32,
    max_threadgroup_memory_bind_count: 31,
    support_dynamic_attribute_stride: true,
};

mod buffer_views;
mod inheritance;
mod records;

/// Each resource ID's records lie in a range of [`records_span`] bytes of their own, every one above
/// every Metal address (below 2^63): `2^63 + id * 2^36` below 2^26, ID 1's from `2^63 + 2^36` and
/// the null ID's below it, the last such range ending at `2^63 + 2^62`, below every higher ID's;
/// `2^63 + 2^62 + id * 2^31` from 2^26, the last ID's range ending at 2^64. Consecutive IDs'
/// ranges lie in order, none reaching the next's start, and touch within each part.
#[test]
fn each_records_id_has_a_range_of_its_own_above_the_metal_addresses() {
    let low = 1u64 << 36;
    assert_eq!(records_address(0), 1 << 63);
    assert_eq!(records_address(1), (1 << 63) + low);
    assert_eq!(records_span(1), low);
    let last_low = RECORDS_HIGH_ID - 1;
    assert_eq!(RECORDS_HIGH_ID, 1 << 26);
    assert_eq!(records_address(last_low) + low, (1 << 63) + (1 << 62));
    let high = 1u64 << 31;
    assert_eq!(records_span(RECORDS_HIGH_ID), high);
    assert_eq!(
        records_address(RECORDS_HIGH_ID),
        (1 << 63) + (1 << 62) + RECORDS_HIGH_ID * high
    );
    assert_eq!(RECORDS_ID_LIMIT, 1 << 31);
    let last = RECORDS_ID_LIMIT - 1;
    assert_eq!(
        u128::from(records_address(last)) + u128::from(records_span(last)),
        1u128 << 64
    );
    for id in [
        1,
        2,
        3,
        65_536,
        70_000,
        last_low,
        RECORDS_HIGH_ID,
        RECORDS_HIGH_ID + 1,
        (1 << 30) + 6,
        last,
    ] {
        let (before, at) = (records_address(id - 1), records_address(id));
        assert!(at > before, "{id}");
        if id != RECORDS_HIGH_ID {
            assert_eq!(at - before, records_span(id), "{id}");
        }
        assert!(before + records_span(id - 1) <= at, "{id}");
    }
}
