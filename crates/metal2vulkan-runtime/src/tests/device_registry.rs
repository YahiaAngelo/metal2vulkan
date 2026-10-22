//! Devices by registry ID (`Device::registry_ids_with`, `Device::with_registry_id`): the IDs are
//! listed with no device made, the device of an ID is the one the other creators make of its
//! physical device (the same device while it lives), and an ID no physical device has is
//! `NoDevice`.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::device::create::live_devices_with;
use crate::{Device, DeviceKind, DeviceOptions, Error, MessageSink};

/// Options of the kind the cfg selects, with a message sink of their own: a group of devices no
/// other test makes (a sink is compared by its identity).
fn options() -> DeviceOptions {
    DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(MessageSink::new(|_| {})),
        ..DeviceOptions::default()
    }
}

#[test]
fn a_device_opens_by_its_registry_id_and_the_ids_make_none() {
    let options = options();
    let ids = Device::registry_ids_with(&options).unwrap();
    assert!(!ids.is_empty(), "no registry ID listed");
    assert_eq!(live_devices_with(&options), 0, "listing made a device");
    // The first ID is the preferred physical device's, of which the runtime makes its default.
    let opened = Device::with_registry_id(ids[0], &options).unwrap();
    assert_eq!(opened.registry_id(), ids[0]);
    assert_eq!(
        live_devices_with(&options),
        1,
        "only the device of the ID is made"
    );
    let default = Device::system_default_with(&options).unwrap();
    assert_eq!(default.registry_id(), ids[0]);
    assert_eq!(
        live_devices_with(&options),
        1,
        "the default is the device opened"
    );
    let all = Device::all_with(&options).unwrap();
    let all: Vec<u64> = all.iter().map(Device::registry_id).collect();
    assert!(
        all.iter().all(|id| ids.contains(id)),
        "{all:x?} among {ids:x?}"
    );
    // Metal never answers 0, so no physical device has it.
    assert!(matches!(
        Device::with_registry_id(0, &options),
        Err(Error::NoDevice)
    ));
}
