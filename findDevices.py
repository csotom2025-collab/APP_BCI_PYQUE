
import hid
import sys

print("Python:", sys.executable)
print("Versión:", sys.version)

devices = hid.enumerate()

print(f"\nDispositivos HID encontrados: {len(devices)}")

for i, dev in enumerate(devices):
    print(f"\n--- Dispositivo {i} ---")
    for key in (
        "vendor_id",
        "product_id",
        "manufacturer_string",
        "product_string",
        "serial_number",
        "interface_number",
        "usage_page",
        "usage",
        "path",
    ):
        print(f"{key}: {dev.get(key)}")
