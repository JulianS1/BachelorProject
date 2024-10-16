import os
import pandas as pd
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
import folium
from folium import Choropleth, CircleMarker
import matplotlib.cm as cm
import matplotlib.colors as colors
import webbrowser as wb


class Mapper:

    def __init__(self) -> None:
        self.data = pd.read_csv("../../data/preprocessed/fauna.csv",
                sep=",",
                encoding="utf-8")
        
    


    def _dms_to_decimal(self, dms):
        """Convert DMS to decimal degrees."""
        dms = dms.replace('"', '').strip()
        print("DMS: ", dms)  # Remove double quotes and trim spaces
        direction = dms[-1:]
        print("Direction: ",direction)
        degrees, minutes_seconds = dms[:-1].split("°")
        print("Minutes & Seconds: ", minutes_seconds)
        minutes, seconds = minutes_seconds.split("'")
        # seconds = seconds_direction[:-1]
        # direction = seconds_direction[-1]

        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        
        # Adjust for southern or western hemispheres
        if direction in ['S', 'W']:
            decimal *= -1
        print(decimal)
        return decimal

    def _make_map(self):
        shapefile_path = '../../data/rawData/za_shp'
        sa_map = gpd.read_file(shapefile_path)

        m = folium.Map(location=[-30.5595, 22.9375], zoom_start=5)

        for _, row in self.data.iterrows():
            print("New Coordinate")
            latitude_decimal = self._dms_to_decimal(row['Latitude'])
            longitude_decimal = self._dms_to_decimal(row['Longitude'])
            folium.Marker(
                location=[latitude_decimal, longitude_decimal],
                popup=folium.Popup(f"""
                    Year: {row['Year']}<br>
                    Location: {row['Location']}<br>
                    Station: {row['Station']}<br>
                    Spionidae: {row['Spionidae']}<br>
                    Port: {row['Port']}
                """, parse_html=True),
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

        # geojson_data = sa_map.__geo_interface__

        # folium.GeoJson(
        #     geojson_data,
        #     name='South Africa',
        #     tooltip=folium.GeoJsonTooltip(fields=['name'])
        # ).add_to(m)

        # folium.LayerControl().add_to(m)

        html_file = 'south_africa_map.html'
        m.save(html_file)
        wb.open('file://' + os.path.realpath(html_file))