"""Registry connecting dataset names to their native-layout adapters."""

try:
    from .datasets.pure import PureAdapter
    from .datasets.ubfc import UbfcAdapter
except ImportError:
    from datasets.pure import PureAdapter
    from datasets.ubfc import UbfcAdapter


DATASET_REGISTRY = {
    "PURE": PureAdapter(),
    "UBFC": UbfcAdapter(),
}


def get_adapter(dataset_name: str):
    name = dataset_name.upper()
    try:
        return DATASET_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {dataset_name!r}. Registered: {available}") from error

