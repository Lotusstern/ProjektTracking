# ProjektTracking

## InfluxDB-Logging aktivieren

1. In `config.yaml` den Block `influx` anpassen. Beispiel:

```yaml
influx:
  enabled: true
  url: http://localhost:8086
  org: example-org
  bucket: detections
  token: <Token mit Schreibrechten>
```

2. Script starten (`python run.py`). Bei aktiven Erkennungen wird pro neuem Inferenz-Ergebnis eine Zeile im Measurement `detections` geschrieben:

```
detections,camera_id=1,label=sports\ ball x=123i,y=240i,r=18i,score=0.87 1699999999999999999
```

3. In InfluxDB/Grafana können Zeitstrahlen oder Scatterplots mit den Feldern `x`, `y`, `r`, `score` nach `camera_id`/`label` gefiltert werden.
