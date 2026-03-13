# ProjektTracking

## InfluxDB-Logging aktivieren

1. Für das Influx-Logging wird das optionale Paket `requests` benötigt (`pip install requests`).

2. In `config.yaml` den Block `influx` anpassen. Beispiel:

```yaml
influx:
  enabled: true
  url: http://localhost:8086
  org: example-org
  bucket: detections
  token: <Token mit Schreibrechten>
```

3. Script starten (`python run.py`). Bei aktiven Erkennungen wird pro neuem Inferenz-Ergebnis eine Zeile im Measurement `detections` geschrieben:

```
detections,camera_id=1,label=sports\ ball x=123i,y=240i,r=18i,score=0.87 1699999999999999999
```

4. In InfluxDB/Grafana können Zeitstrahlen oder Scatterplots mit den Feldern `x`, `y`, `r`, `score` nach `camera_id`/`label` gefiltert werden.

## Grafana + InfluxDB Integration (Schritt-für-Schritt)

### 1) Vorbereitungen prüfen

- InfluxDB läuft und ist erreichbar (z. B. `http://localhost:8086`).
- Das Logging in `config.yaml` ist aktiviert (`influx.enabled: true`).
- Der Token hat Schreibrechte auf den Bucket (`Tracking`/`detections`) und für Grafana idealerweise auch Leserechte.

### 2) Datenfluss verifizieren

1. Tracker starten (`python run.py`) und kurz laufen lassen.
2. In InfluxDB Data Explorer prüfen, ob im Measurement `detections` Daten ankommen.
3. Erwartete Struktur:
   - **Tags:** `camera_id`, `label`
   - **Fields:** `x`, `y`, `r`, `score`
   - **Timestamp:** automatisch beim Schreiben in Nanosekunden

### 3) InfluxDB als Data Source in Grafana anlegen

In Grafana unter **Connections → Data sources → Add data source → InfluxDB**:

- Query Language: **Flux** (für flexible Auswertungen)
- URL: `http://localhost:8086`
- Organization: wie in `config.yaml` (`Berufskolleg`)
- Token: Influx API-Token
- Default Bucket: `Tracking`

Danach **Save & test**.

### 4) Dashboard mit Kern-Panels erstellen

Empfohlene Panels:

1. **Erkennungen pro Zeitfenster (Time series)**
   - Zeigt Aktivität (Spitzen/Leerlauf).
2. **Score-Verlauf (Time series)**
   - Qualität/Confidence über Zeit.
3. **Positionen als Scatter/XY**
   - Wo wurde erkannt (`x` gegen `y`).
4. **Heatmap / 2D-Häufigkeit**
   - Hotspots im Kamerabild.
5. **Top Labels / Kamera-Filter (Table/Bar)**
   - Welche Objekte/Kameras dominieren.

### 5) Nützliche Dashboard-Variablen

Unter **Dashboard settings → Variables**:

- `camera_id` (Tag-Filter)
- `label` (Tag-Filter)

So kann man alle Panels dynamisch nach Kamera und erkanntem Objekttyp filtern.

---

## Beispiel-Flux-Queries

> Hinweis: Bucket-Name ggf. auf `detections` anpassen, falls abweichend konfiguriert.

### A) Erkennungen pro Minute

```flux
from(bucket: "Tracking")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "detections")
  |> filter(fn: (r) => r._field == "score")
  |> filter(fn: (r) => r.camera_id == "${camera_id}" or "${camera_id}" == "all")
  |> filter(fn: (r) => r.label == "${label}" or "${label}" == "all")
  |> aggregateWindow(every: 1m, fn: count, createEmpty: false)
  |> yield(name: "detections_per_min")
```

### B) Score-Verlauf

```flux
from(bucket: "Tracking")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "detections")
  |> filter(fn: (r) => r._field == "score")
  |> filter(fn: (r) => r.camera_id == "${camera_id}" or "${camera_id}" == "all")
  |> filter(fn: (r) => r.label == "${label}" or "${label}" == "all")
  |> yield(name: "score_over_time")
```

### C) XY-Scatter (Positionen)

```flux
x = from(bucket: "Tracking")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "detections" and r._field == "x")
  |> filter(fn: (r) => r.camera_id == "${camera_id}" or "${camera_id}" == "all")
  |> filter(fn: (r) => r.label == "${label}" or "${label}" == "all")
  |> keep(columns: ["_time", "_value", "camera_id", "label"])
  |> rename(columns: {_value: "x"})

y = from(bucket: "Tracking")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "detections" and r._field == "y")
  |> filter(fn: (r) => r.camera_id == "${camera_id}" or "${camera_id}" == "all")
  |> filter(fn: (r) => r.label == "${label}" or "${label}" == "all")
  |> keep(columns: ["_time", "_value"])
  |> rename(columns: {_value: "y"})

join(tables: {x: x, y: y}, on: ["_time"])
  |> yield(name: "xy_points")
```

### D) Heatmap-Vorbereitung (Binning)

```flux
import "math"

data = from(bucket: "Tracking")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "detections")
  |> filter(fn: (r) => r._field == "x" or r._field == "y")
  |> pivot(rowKey: ["_time", "camera_id", "label"], columnKey: ["_field"], valueColumn: "_value")
  |> filter(fn: (r) => r.camera_id == "${camera_id}" or "${camera_id}" == "all")
  |> filter(fn: (r) => r.label == "${label}" or "${label}" == "all")
  |> map(fn: (r) => ({
      r with
      x_bin: int(v: math.floor(x: float(v: r.x) / 32.0)),
      y_bin: int(v: math.floor(x: float(v: r.y) / 24.0))
    }))
  |> group(columns: ["x_bin", "y_bin"])
  |> count(column: "x")

data |> yield(name: "heatmap_bins")
```

---

## Troubleshooting

- **Keine Daten in Grafana:**
  - Prüfen, ob `run.py` läuft und tatsächlich Erkennungen stattfinden.
  - Influx URL/Org/Bucket/Token in `config.yaml` kontrollieren.
- **Data Source Test schlägt fehl:**
  - Netzwerk/Port (`8086`) und Token-Rechte prüfen.
- **Leeres Panel bei Filtern:**
  - Variable-Werte kontrollieren (`all` korrekt behandelt?).
  - Zeitbereich im Dashboard vergrößern.
- **Heatmap wirkt “verschoben”:**
  - X/Y-Achsenrichtung im Panel prüfen (Y-Achse bei Bilddaten oft invertiert).
