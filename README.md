# Instagram Location Dashboard
An online dashboard to search for Instagram posts in a certain location and plot those on a map. 
- Also allows the user to upload their own geolocation CSV file to plot. 
- This is an expansion of the ´instagram-location-search´ repo found [here](https://github.com/bellingcat/instagram-location-search)

## Upload File
1. Click upload file
2. Choose CSV file with GPS coordinates columns

## How to contribute to this

Assumes you have Python3.11 and its ´virtualenv´ pip library
1. Clone the repo
2. Run `python3.11 -m venv env`
3. Run `source env/bin/activate`
4. Run `streamlit run app/main.py`

### Testing
- Create a file called `.env` in the root directory and create a variable called `INSTAGRAM_COOKIES`


## Versioning
### Tags

## Structure
Using this file structure for Streamlit: https://github.com/ash2shukla/streamlit-heroku/blob/master/app/main.py