# Setup Guide

The wind-field API runs in Google Colab. Nothing has to be installed on your computer.

## 1. Make two free accounts (only once)

| Account | Why | Sign up |
|---|---|---|
| Google Earth Engine | downloads the Sentinel-1 images | https://earthengine.google.com/ → *Get Started* → register a **non-commercial** Cloud project. Note the **project id** (e.g. `my-wind-project`). |
| Copernicus Marine | reference wind (decides the correct direction out of two opposite ones) | https://marine.copernicus.eu/ → *Register*. Note your **username** and **password**. |

## 2. Run the API

1. Open the repository page and click **Open in Colab**.
2. **Setup cell:** type your Earth Engine project id in `EE_PROJECT` and press ▶.
   - A Google window opens → choose your account → **Allow**.
   - Type your Copernicus Marine username and password when asked.
   - Wait for `Setup complete.` (about 1–2 minutes).
3. **Step 1:** press ▶. A map appears. Click the square tool (◻) and draw a box on the sea.
   The box cannot be bigger than 50 km × 50 km.
4. **Step 2:** choose the start date and end date, press ▶. Wait a few minutes. The wind maps appear below the cell.
5. **Step 3:** press ▶ to download all results as a zip file.

To try another area or dates: draw a new box in Step 1 and run Step 2 again. Setup is needed only once per session.

## Common problems

| Message | What to do |
|---|---|
| `Enter your Earth Engine Cloud project id` | Type the project id in `EE_PROJECT` and run Setup again. |
| `No area of interest selected` | Draw the box in Step 1 first, then run Step 2. |
| `Box is ... km; the maximum is 50 x 50 km` | Draw a smaller box. |
| `no Sentinel-1 IW acquisition found` | Use a longer date range (at least 2 weeks). The message lists the nearest dates. |
| `WARNING: weak SAR texture` on the map | Wind is weak or there is rain; trust only the black arrows. |
| Old code still runs after an update | *Runtime → Restart session*, then run Setup again. |
