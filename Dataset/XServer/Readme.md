# Readme

## Data Format

### Lidar

```json
{
  "timestamp": "2023-07-10T16:37:41:731383",
  "position": [1.0, 2.0, 3.0],
  "orientation": [1.56, 1.43, 2.34, 0.67]
}
```



### BLE Gateway

Each BLE gateway contains two queues, in which the tagid is different (1 or 2).

**Queue1**

```json
{
  "timestamp": "2023-07-10T16:37:41:731383",
  "tagid":"1",
  "frequency":920e6,
  "sequence": "110000011",
  "rssi": -61,
	"samples": [1.3, 4.1, 5.1, 2.3]
  "timestamp": "2023-07-10T16:37:41:751383"
}
```

