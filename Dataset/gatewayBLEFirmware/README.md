# mosquitto setup

## mac 
```bash
    brew install mosquitto
    # run mosquitto
    /usr/local/sbin/mosquitto
```

## ubuntu

```bash
    sudo apt install -y mosquitto mosquitto-clients
    # run mosquitto
    mosquitto
```
```bash
# check serial
ls -l /dev/
ls /dev/tty.*
```

# firmware setup

```bash
cd ~/export/app/bluetooth/example_host/bt_aoa_host_locator
make
cd exe
./bt_aoa_host_locator -u /dev/ttyACM0 -c ../config/locator_config.json
```