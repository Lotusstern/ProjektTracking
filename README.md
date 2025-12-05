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

## Nächste Schritte nach dem Patch

1. **InfluxDB bereitstellen**
   - Lokal: InfluxDB 2.x starten (`docker run -p 8086:8086 influxdb:2`).
   - Alternativ einen bestehenden Server/Bucket nutzen.

2. **Organisation/Bucket/Token anlegen**
   - Im InfluxDB-UI (http://localhost:8086) eine Organisation (falls nicht vorhanden) und einen Bucket `detections` anlegen.
   - Unter „API Tokens“ einen **Write Token** für den Bucket erstellen und in `config.yaml` eintragen.

3. **`config.yaml` ausfüllen**
   - `enabled: true` setzen, URL/Org/Bucket/Token eintragen.
   - Optional: `camera_id` setzen, falls mehrere Kameras genutzt werden.

4. **Testen, ob das Schreiben klappt**
   - `python run.py` starten und eine Erkennung provozieren.
   - In InfluxDB unter „Data Explorer“ Measurement `detections` auswählen und prüfen, ob neue Punkte erscheinen (Tags: `camera_id`, `label`; Felder: `x`, `y`, `r`, `score`).

5. **Grafana-Dashboard bauen (optional)**
   - In Grafana eine InfluxDB-Data-Source auf den Bucket legen.
   - Panels anlegen, z. B.:
     - Zeitstrahl: Anzahl Erkennungen pro Minute nach `label`/`camera_id`.
     - Scatter: `score` über Zeit, farblich nach `label`.
     - Aktuelle Position: Tabelle/Stat mit `x`, `y`, `r` der letzten Erkennung je Kamera.

6. **Produktionshärtung (optional)**
   - Retention Policy/Bucket-Aufbewahrung festlegen (z. B. 30 Tage Rohdaten).
   - Batching aktivieren (falls du später mehr Traffic hast): mehrere Zeilen sammeln und gemeinsam senden.
