# BLE Data Collection

## BLE Anchor

BRD4109 antenna array: NCP locator

array (locator) -> board -> raspiey (analyzer)

```bash
 # run mosquitto
 /usr/local/sbin/mosquitto
# run analyzer


```

## BLE tag

2 tags (aoa tag)

## Ground Truth

Lidar (LIO-SAM, https://github.com/TixiaoShan/LIO-SAM)







## BLE Data Format

The dataset is represented as a JSON object with the following structure:

```json
{
    "tagID":"1",
    "logTime":"2021-06-10 16:37:41:731383",
    "phyTime":"2021-07-10 16:37:42:885076",
    "savedTime":"2021-07-10 16:37:49:275",
    "position": [-0.52, 0.19, -1.27],
    "xServer":{
        "frequency":920e6,
        "gateways":{
            "gateway1":{
                "version":v1,
                "position":[p1,p2,p3,p4],
                "samples":[p1,p1,....,p16],
                "phaseOffset":[o1,o2,....,o16],
                "rssi":[r1,r2,..., r16],
                "createdTime": us
            },
            "gateway2":{
               "version":"v1",
                "position":[p1,p2,p3,p4],
                "phase":[p1,p1,....,p16],
                "phaseOffset":[o1,o2,....,o16],
                "rss":[r1,r2,..., r16],
                "createdTime": "2021-07-10 16:37:41:732359"
            }
        }
    }
}
```

Here is the explanation of each field:

- **tagId**: the ID of the RFID tag
- **logTime**: the logical time logged (gateways and lidar aligned time) after the queue alignment by the server.
- **phyTime**: The physical timestamp produced by the Xserver.
- **saveTime**: The timestamp of when this data is stored into the database.
- **position**: The computed position as determined by the Xserver.
- **truth**: The position computed by the OptiTrack system.
- **xServer**: This section merges multiple message queues and includes:
  - **frequency**: The carrier frequency of the continuous wave (CW).
  - **gateways**: Information about the gateways at a single timestamp, each identified by a unique ID (e.g., gateway1, gateway2).
    - **version**: The hardware version of the gateway.
    - **position**: The geographical position of the gateway.
    - **phase**: The calibrated phase values.
    - **phaseOffset**: The offset used for phase calibration.
    - **rss**: The RSS for each element in the gateway
    - **createTime**:The time at which the gateway estimated the phase.
