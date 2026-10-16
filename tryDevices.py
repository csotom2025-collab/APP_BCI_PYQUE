
import hid
import time

VID = 0x21A1
PID = 0x0001

devices = hid.enumerate(VID, PID)

print(f"Interfaces Emotiv encontradas: {len(devices)}")

for info in devices:
    print("\n" + "=" * 60)
    print("Producto:", info.get("product_string"))
    print("Interfaz:", info.get("interface_number"))
    print("Usage page:", hex(info.get("usage_page", 0)))
    print("Path:", info.get("path"))

    device = hid.device()

    try:
        device.open_path(info["path"])
        device.set_nonblocking(0)
        print("Dispositivo abierto correctamente.")

        for size in (32, 64):
            print(f"\nProbando lecturas de {size} bytes:")

            for i in range(3):
                start = time.time()
                data = device.read(size, timeout_ms=1500)
                elapsed = time.time() - start

                print(
                    f"Intento {i + 1}: "
                    f"{len(data)} bytes recibidos, "
                    f"tiempo={elapsed:.2f}s"
                )

                if data:
                    print("Primeros bytes:", list(data[:12]))

    except Exception as exc:
        print("ERROR:", repr(exc))

    finally:
        try:
            device.close()
        except Exception:
            pass

print("\nPrueba terminada.")
