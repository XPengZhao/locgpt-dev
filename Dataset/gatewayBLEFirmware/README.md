# mosquitto setup

## mac 
```bash
    # install mosquitto
    brew install mosquitto
    # run mosquitto
    /usr/local/sbin/mosquitto
```

## ubuntu

```bash
    # install mosquitto
    sudo apt install -y mosquitto mosquitto-clients
    # run mosquitto
    sudo systemctl start mosquitto
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
    # run firmware with seril port /dev/ttyACM0
    ./bt_aoa_host_locator -u /dev/ttyACM0 -c ../config/locator_config.json
```