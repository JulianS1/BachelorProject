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
        # print("DMS: ", dms)  # Remove double quotes and trim spaces
        direction = dms[-1:]
        # print("Direction: ",direction)
        degrees, minutes_seconds = dms[:-1].split("°")
        # print("Minutes & Seconds: ", minutes_seconds)
        minutes, seconds = minutes_seconds.split("'")
        # seconds = seconds_direction[:-1]
        # direction = seconds_direction[-1]

        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        
        # Adjust for southern or western hemispheres
        if direction in ['S', 'W']:
            decimal *= -1
        # print(decimal)
        return decimal
    
    def _get_color(self, spionidae_count):
            """Return a color based on the global variance of the Spionidae count."""
            if spionidae_count < self.spionidae_mean - self.spionidae_std:
                return 'red'  # Low values
            elif self.spionidae_mean - self.spionidae_std <= spionidae_count < self.spionidae_mean:
                return 'orange'  # Below average
            elif self.spionidae_mean <= spionidae_count < self.spionidae_mean + self.spionidae_std:
                return 'yellow'  # Above average
            else:
                return 'green'  # High values

    def _make_map(self):
        shapefile_path = '../../data/rawData/za_shp'
        sa_map = gpd.read_file(shapefile_path)

        m = folium.Map(location=[-30.5595, 22.9375], zoom_start=5)

        # Find average of loacations
        # print("Showing latitude in decimal: \n", self.data["Latitude"].apply(self._dms_to_decimal))
        
        self.data["Latitude"] = self.data["Latitude"].apply(self._dms_to_decimal)
        self.data["Longitude"] = self.data["Longitude"].apply(self._dms_to_decimal)

        # for _, row in self.data.iterrows():
        average_locations = self.data.groupby('Station(Newnumber)').agg({
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Year': 'first',  # Example: taking the first year entry as an example
            'Location': 'first',
            'SQILowerlimit': 'mean',
            'Spionidae': 'mean',  # Similar for Spionidae
            'Port': 'first'  # Take the first port as an example
        }).reset_index()
        self.spionidae_mean = average_locations['Spionidae'].mean()
        self.spionidae_std = average_locations['Spionidae'].std()
        

        for _, row in average_locations.iterrows():
            marker_color = self._get_color(row['Spionidae'])
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(f"""
                    Station: {row['Station(Newnumber)']}\n
                    Average Latitude: {row['Latitude']}\n
                    Average Longitude: {row['Longitude']}\n
                    Year: {row['Year']}\n
                    Location: {row['Location']}\n
                    Average SQI: {row['SQILowerlimit']}\n
                    Average Spionidae: {row['Spionidae']}\n
                    Port: {row['Port']}
                """, parse_html=True),
                icon=folium.Icon(color=marker_color, icon='info-sign')
            ).add_to(m)


        html_file = 'south_africa_map.html'
        m.save(html_file)
        wb.open('file://' + os.path.realpath(html_file))