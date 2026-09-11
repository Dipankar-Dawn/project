import os
import ee
import geemap


# =========================================================
# EARTH ENGINE CONFIGURATION
# =========================================================

EE_PROJECT = "moilai"

EE_KEY_FILE = "/etc/secrets/earthengine-service-account.json"


# =========================================================
# INITIALIZE EARTH ENGINE
# =========================================================

def initialize_earth_engine():

    try:

        if os.path.exists(EE_KEY_FILE):

            # Render / Production
            credentials = ee.ServiceAccountCredentials(
                None,
                key_file=EE_KEY_FILE
            )

            ee.Initialize(
                credentials=credentials,
                project=EE_PROJECT
            )

            print(
                "Earth Engine initialized using service account."
            )

            return True

        else:

            # Local development
            ee.Initialize(
                project=EE_PROJECT
            )

            print(
                "Earth Engine initialized using local credentials."
            )

            return True

    except Exception as e:

        print(
            "Earth Engine initialization failed:"
        )

        print(e)

        return False


# =========================================================
# GENERATE MANGANESE MAP
# =========================================================

def generate_manganese_map(
    west,
    south,
    east,
    north,
    output_file
):

    try:

        # -------------------------------------------------
        # ROI
        # -------------------------------------------------

        roi = ee.Geometry.Rectangle([
            west,
            south,
            east,
            north
        ])


        # -------------------------------------------------
        # CREATE MAP
        # -------------------------------------------------

        Map = geemap.Map()


        Map.centerObject(
            roi,
            11
        )


        # -------------------------------------------------
        # ROI LAYER
        # -------------------------------------------------

        Map.addLayer(
            roi,
            {
                "color": "red"
            },
            "Selected Area"
        )


        # -------------------------------------------------
        # SENTINEL-2
        # -------------------------------------------------

        sentinel = (
            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )
            .filterBounds(roi)
            .filterDate(
                "2025-01-01",
                "2026-06-30"
            )
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    15
                )
            )
        )


        # -------------------------------------------------
        # MEDIAN IMAGE
        # -------------------------------------------------

        image = (
            sentinel
            .median()
            .clip(roi)
        )


        # -------------------------------------------------
        # SENTINEL-2 RGB
        # -------------------------------------------------

        Map.addLayer(
            image,
            {
                "bands": [
                    "B4",
                    "B3",
                    "B2"
                ],
                "min": 0,
                "max": 3000
            },
            "Sentinel-2 RGB"
        )


        # -------------------------------------------------
        # FERROUS IRON RATIO
        # -------------------------------------------------

        ferrous_ratio = (
            image
            .select("B11")
            .divide(
                image.select("B8")
            )
            .rename("ferrous_ratio")
        )


        # -------------------------------------------------
        # CLAY MINERAL INDEX
        # -------------------------------------------------

        clay_index = (
            image
            .select("B11")
            .divide(
                image.select("B12")
            )
            .rename("clay_index")
        )


        # -------------------------------------------------
        # IRON OXIDE RATIO
        # -------------------------------------------------

        iron_oxide_ratio = (
            image
            .select("B4")
            .divide(
                image.select("B2")
            )
            .rename("iron_oxide_ratio")
        )


        # -------------------------------------------------
        # MANGANESE SPECTRAL PROXY
        # -------------------------------------------------

        mn_proxy = (
            image
            .select("B11")
            .add(
                image.select("B4")
            )
            .divide(
                image.select("B8")
                .add(
                    image.select("B2")
                )
            )
            .rename("mn_proxy")
        )


        # -------------------------------------------------
        # CALCULATE PERCENTILES
        # -------------------------------------------------

        stats = mn_proxy.reduceRegion(
            reducer=ee.Reducer.percentile(
                [
                    5,
                    25,
                    50,
                    75,
                    95
                ]
            ),
            geometry=roi,
            scale=20,
            maxPixels=1e9
        )


        # -------------------------------------------------
        # GET 95TH PERCENTILE
        # -------------------------------------------------

        threshold = stats.get(
            "mn_proxy_p95"
        ).getInfo()


        print(
            "Mn Proxy 95th percentile:",
            threshold
        )


        # -------------------------------------------------
        # MANGANESE ANOMALY
        # -------------------------------------------------

        mn_anomaly = mn_proxy.gt(
            ee.Number(threshold)
        )


        Map.addLayer(
            mn_anomaly.selfMask(),
            {
                "palette": [
                    "red"
                ]
            },
            "Mn Proxy > 95th Percentile"
        )


        # -------------------------------------------------
        # NDVI
        # -------------------------------------------------

        ndvi = (
            image
            .select("B8")
            .subtract(
                image.select("B4")
            )
            .divide(
                image.select("B8")
                .add(
                    image.select("B4")
                )
            )
            .rename("NDVI")
        )


        Map.addLayer(
            ndvi,
            {
                "min": -0.2,
                "max": 0.8,
                "palette": [
                    "brown",
                    "yellow",
                    "green"
                ]
            },
            "NDVI"
        )


        # -------------------------------------------------
        # VEGETATION FILTER
        # -------------------------------------------------

        non_vegetated = ndvi.lt(
            0.4
        )


        mn_anomaly_filtered = (
            mn_anomaly
            .And(non_vegetated)
        )


        Map.addLayer(
            mn_anomaly_filtered.selfMask(),
            {
                "palette": [
                    "red"
                ]
            },
            "Mn Anomaly + NDVI Filter"
        )


        # -------------------------------------------------
        # FALSE COLOR
        # -------------------------------------------------

        Map.addLayer(
            image,
            {
                "bands": [
                    "B11",
                    "B8",
                    "B4"
                ],
                "min": 0,
                "max": 3500
            },
            "SWIR-NIR-Red False Color"
        )


        # -------------------------------------------------
        # SAVE MAP
        # -------------------------------------------------

        Map.to_html(
            output_file
        )


        print(
            "✓ Sentinel-2 manganese map generated."
        )


        return {

            "success": True,

            "threshold": float(
                threshold
            ),

            "output_file": output_file

        }


    except Exception as e:

        print(
            "✗ Sentinel-2 map generation error:"
        )

        print(e)


        return {

            "success": False,

            "error": str(e)

        }