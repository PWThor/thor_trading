import pandas as pd
import os

# Define the paths to the CSV files
base_path = "E:/Projects/thor_trading/data/raw/"
city_files = {
    "Chicago": "Chicago_41_878114_-87_629798_694224000_1744156799_681c29a82414a80008e8c060.csv",
    "Cushing": "Cushing_35_985064_-96_76697_694224000_1744156799_681c29a82414a80008e8c060.csv",
    "Houston": "Houston_29_760077_-95_370111_694224000_1744156799_681c29a82414a80008e8c060.csv",
    "NYC": "New_York_40_712775_-74_005973_694224000_1744156799_681c29a82414a80008e8c060.csv"
}

# Output path for the combined CSV
output_path = "E:/Projects/thor_trading/data/raw/weather/weather_data.csv"

# Combine the CSV files
combined_data = []
for city, file_name in city_files.items():
    file_path = os.path.join(base_path, file_name)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract the datetime from dt_iso (keeping hourly granularity)
    df['Date'] = pd.to_datetime(df['dt_iso'].str.split(' ').str[0] + ' ' + df['dt_iso'].str.split(' ').str[1])
    
    # Use city_name as Region
    df['Region'] = df['city_name']
    
    # Convert temperature from Celsius to Fahrenheit for AvgTemperature
    df['AvgTemperature'] = (df['temp'] * 9/5) + 32
    
    # Use rain_1h as Precipitation, default to 0 if missing
    df['Precipitation'] = df['rain_1h'].fillna(0)
    
    # Use wind_speed as WindSpeed
    df['WindSpeed'] = df['wind_speed']
    
    # Use snow_1h as Snowfall, default to 0 if missing
    df['Snowfall'] = df['snow_1h'].fillna(0)
    
    # Use humidity as Humidity
    df['Humidity'] = df['humidity']
    
    # Use clouds_all as CloudCoverage
    df['CloudCoverage'] = df['clouds_all']
    
    # Select the required columns
    df = df[['Date', 'Region', 'AvgTemperature', 'Precipitation', 'WindSpeed', 'Snowfall', 'Humidity', 'CloudCoverage']]
    
    # Append to the combined data
    combined_data.append(df)

# Concatenate all DataFrames
if combined_data:
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Ensure Date is in datetime format
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # Sort by Date and Region
    combined_df = combined_df.sort_values(['Date', 'Region'])
    
    # Save to the output path
    combined_df.to_csv(output_path, index=False)
    print(f"Combined weather data saved to {output_path}")
else:
    print("No data was combined. Please check the input files.")