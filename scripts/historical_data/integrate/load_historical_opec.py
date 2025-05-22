import os
import logging
import datetime
import re
import subprocess
import psycopg2
import tabula
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np

# Set up logging
log_dir = "E:/Projects/thor_trading/outputs/logs/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'load_historical_opec.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database connection parameters
DB_PARAMS = {
    "dbname": "trading_db",
    "user": "postgres",
    "password": "Makingmoney25!",
    "host": "localhost",
    "port": "5432"
}

# Directory containing OPEC MOMR PDFs
PDF_DIR = "E:/Projects/thor_trading/data/raw/opec/"

# Preprocessing for OCR
def preprocess_image(image):
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
    except Exception as e:
        logging.error(f"Error in preprocess_image: {str(e)}")
        return np.array(image)

# Extract text using OCR
def extract_text_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        text = ""
        for image in images:
            processed_image = preprocess_image(image)
            custom_config = r'--oem 3 --psm 6'
            text += pytesseract.image_to_string(processed_image, config=custom_config) + "\n"
        return text
    except Exception as e:
        logging.error(f"OCR error for {pdf_path}: {str(e)}")
        return ""

# Extract text using PyPDF2
def extract_text_with_pypdf2(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logging.error(f"PyPDF2 error for {pdf_path}: {str(e)}")
        return ""

# Parse metrics from extracted text
def parse_metrics_from_text(text):
    metrics = {}
    patterns = {
        "OPEC_Crude_Production": r"(?<!\w)OPEC Crude Production[^0-9]*(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Saudi_Arabia": r"(?<!\w)(Saudi Arabia|Saudi)\s*(?:crude production)?[^0-9]{0,30}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Iran": r"(?<!\w)Iran\s*(?:crude production)?[^0-9]{0,30}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Iraq": r"(?<!\w)Iraq\s*(?:crude production)?[^0-9]{0,30}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Kuwait": r"(?<!\w)Kuwait\s*(?:crude production)?[^0-9]{0,20}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Venezuela": r"(?<!\w)Venezuela\s*(?:crude production)?[^0-9]{0,30}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Nigeria": r"(?<!\w)Nigeria\s*(?:crude production)?[^0-9]{0,30}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Algeria": r"(?<!\w)Algeria\s*(?:crude production)?[^0-9]{0,30}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "OPEC_Crude_Production_Angola": r"(?<!\w)Angola\s*(?:crude production)?[^0-9]{0,30}(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])",
        "World_Oil_Demand": r"(?<!\w)World Oil Demand[^0-9]*(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])"
    }
    for metric, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                # Log the full match and groups for debugging
                logging.info(f"Pattern match for {metric}: Full match='{match.group(0)[:100]}...', Group 1='{match.group(1)}'")
                # Group 1 should be the numeric value
                value = float(match.group(1))
                # Adjusted validation to capture more metrics
                if metric == "World_Oil_Demand" and not (50 <= value <= 200):  # Tightened lower bound
                    logging.warning(f"Invalid World_Oil_Demand value {value} - skipping")
                    continue
                if metric.startswith("OPEC_Crude_Production_") and metric != "OPEC_Crude_Production" and not (0 <= value <= 15):
                    logging.warning(f"Invalid {metric} value {value} - skipping")
                    continue
                if metric == "OPEC_Crude_Production" and not (10 <= value <= 50):
                    logging.warning(f"Invalid OPEC_Crude_Production value {value} - skipping")
                    continue
                metrics[metric] = {"value": value, "unit": "million barrels per day"}
                logging.info(f"Successfully matched {metric}: {value} mb/d")
            except (ValueError, IndexError) as e:
                logging.warning(f"Failed to parse value for {metric}: {str(e)}, matched text='{match.group(0)[:100]}...'")
                continue
        # If no metrics were found for this pattern, log a warning
        if metric not in metrics:
            logging.warning(f"No valid metrics found for {metric} in this report")
    return metrics

# Extract tables using tabula-py
def extract_tables(pdf_path):
    try:
        tables = tabula.read_pdf(pdf_path, pages="all", lattice=True, guess=True, output_format="json")
        return tables
    except subprocess.CalledProcessError as e:
        logging.error(f"Tabula-py error for {pdf_path}: {e.output.decode('utf-8', errors='ignore')}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in tabula-py for {pdf_path}: {str(e)}")
        return None

# Parse metrics from tables
def parse_metrics_from_tables(tables):
    metrics = {}
    if not tables:
        return metrics
    for table in tables:
        data = table.get("data", [])
        for row in data:
            if isinstance(row, list) and len(row) >= 2:
                text = " ".join([str(cell.get("text", "")) for cell in row])
                for metric in [
                    "OPEC_Crude_Production", "OPEC_Crude_Production_Saudi_Arabia", "OPEC_Crude_Production_Iran",
                    "OPEC_Crude_Production_Iraq", "OPEC_Crude_Production_Kuwait", "OPEC_Crude_Production_Venezuela",
                    "OPEC_Crude_Production_Nigeria", "OPEC_Crude_Production_Algeria", "OPEC_Crude_Production_Angola",
                    "World_Oil_Demand"
                ]:
                    country_name = metric.replace("OPEC_Crude_Production_", "").replace("_", " ")
                    pattern = rf"(?<!\w){re.escape(country_name)}\s*(?:crude production)?[^0-9]*?(\d+\.\d+|\d+\b)\s*(?:mb/d|million barrels per day)?(?=\s|$|\n|[.,])"
                    value_match = re.search(pattern, text, re.IGNORECASE)
                    if value_match:
                        try:
                            value = float(value_match.group(1))
                            # Adjusted validation
                            if metric == "World_Oil_Demand" and not (50 <= value <= 200):
                                logging.warning(f"Invalid World_Oil_Demand value {value} in table - skipping")
                                continue
                            if metric.startswith("OPEC_Crude_Production_") and metric != "OPEC_Crude_Production" and not (0 <= value <= 15):
                                logging.warning(f"Invalid {metric} value {value} in table - skipping")
                                continue
                            if metric == "OPEC_Crude_Production" and not (10 <= value <= 50):
                                logging.warning(f"Invalid OPEC_Crude_Production value {value} in table - skipping")
                                continue
                            metrics[metric] = {"value": value, "unit": "million barrels per day"}
                            logging.info(f"Matched {metric}: {value} mb/d from table row: {text[:100]}...")
                        except ValueError as e:
                            logging.warning(f"Failed to parse value for {metric} in table: {str(e)}, text='{text[:100]}...'")
                            continue
    return metrics

# Insert data into the database
def insert_data(conn, report_date, metric, value, unit):
    try:
        cursor = conn.cursor()
        # Check for duplicates
        cursor.execute("""
            SELECT 1 FROM opec_data 
            WHERE report_date = %s AND metric = %s AND value = %s AND unit = %s
        """, (report_date, metric, value, unit))
        if cursor.fetchone():
            logging.info(f"Skipping duplicate OPEC data for {metric} on {report_date}")
            return
        
        # Insert the record
        cursor.execute("""
            INSERT INTO opec_data (report_date, metric, value, unit) 
            VALUES (%s, %s, %s, %s)
        """, (report_date, metric, value, unit))
        conn.commit()
        logging.info(f"Inserted OPEC data for {metric} on {report_date}")
    except Exception as e:
        logging.error(f"Error inserting data for {metric} on {report_date}: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()

# Clear the opec_data table
def clear_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE opec_data")
        conn.commit()
        logging.info("Cleared opec_data table")
    except Exception as e:
        logging.error(f"Error clearing opec_data table: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()

# Main function to process PDFs
def main():
    print("Starting the OPEC data loading script...")
    logging.info("Starting the OPEC data loading script...")
    
    # Check PDF directory
    if not os.path.exists(PDF_DIR):
        error_msg = f"PDF directory does not exist: {PDF_DIR}"
        print(error_msg)
        logging.error(error_msg)
        return
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        error_msg = f"No PDF files found in {PDF_DIR}"
        print(error_msg)
        logging.error(error_msg)
        return
    print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")
    logging.info(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")

    # Connect to the database
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        print("Connected to the database successfully.")
        logging.info("Database connection successful")
    except Exception as e:
        error_msg = f"Failed to connect to database: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return

    try:
        # Clear the table
        clear_table(conn)

        # Process each year and month from 2001 to 2025
        for year in range(2001, 2026):
            for month in range(1, 13):
                # Skip future dates
                report_date = datetime.date(year, month, 1)
                if report_date > datetime.date.today():
                    break

                # Construct possible PDF filenames
                month_name = report_date.strftime("%B").lower()
                possible_filenames = [
                    f"momr-{month_name}-{year}.pdf",
                    f"momr-{month_name}-{year}-1.pdf",
                    f"momr-{month_name}-{year}-2.pdf"
                ]

                all_metrics = {}
                files_processed = []
                for filename in possible_filenames:
                    candidate_path = os.path.join(PDF_DIR, filename)
                    if os.path.exists(candidate_path):
                        logging.info(f"Found file for {year}-{month:02d}: {candidate_path}")
                        files_processed.append(candidate_path)
                        # Extract metrics from this file
                        logging.info(f"Extracting OPEC data for {year}-{month:02d} from {candidate_path}")
                        tables = extract_tables(candidate_path)
                        if tables:
                            metrics = parse_metrics_from_tables(tables)
                        else:
                            logging.info(f"Error extracting tables for {year}-{month:02d}: Falling back to PyPDF2")
                            text = extract_text_with_pypdf2(candidate_path)
                            if text:
                                metrics = parse_metrics_from_text(text)
                            else:
                                logging.info(f"Falling back to OCR for {candidate_path}")
                                text = extract_text_with_ocr(candidate_path)
                                metrics = parse_metrics_from_text(text)

                        # Log the number of metrics extracted
                        logging.info(f"Extracted {len(metrics)} metrics from {candidate_path}")
                        if not metrics:
                            logging.warning(f"No metrics extracted for {year}-{month:02d} from {candidate_path}")

                        # Combine metrics from all files for this month
                        all_metrics.update(metrics)

                if not files_processed:
                    logging.warning(f"OPEC MOMR file for {year}-{month:02d} not found in {PDF_DIR}")
                    continue

                # Log the total metrics for this month
                logging.info(f"Total {len(all_metrics)} unique metrics extracted for {year}-{month:02d} from {len(files_processed)} files")

                # Insert extracted metrics into the database
                for metric, data in all_metrics.items():
                    logging.info(f"Inserting OPEC data for {metric} on {report_date}: Value={data['value']}, Unit={data['unit']}")
                    insert_data(conn, report_date, metric, data["value"], data["unit"])

        # Close the database connection
        conn.close()
        print("Script completed successfully.")
        logging.info("Script completed successfully.")
    except Exception as e:
        error_msg = f"Script failed with error: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()