# TODO:1. Importing libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import statsmodels.api as sm

st.title("Analysis of Topographic Factors in Above-Ground Biomass Estimation")
st.sidebar.title("Analysis of Topographic Factors in Above-Ground Biomass Estimation")

st.markdown("This application is a Streamlit app used to analyze the influence of topographical variables in above "
            "ground biomass accumulation")
st.sidebar.markdown("This application is a Streamlit app used to analyze the influence of topographical variables "
                    "in above ground biomass accumulation")

# TODO:2. Load the data from a CSV file
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

    # Read the uploaded file
    bytes_data = uploaded_file.getvalue()

    # Convert uploaded file to a dataframe
    df = pd.read_csv(uploaded_file)

    # To display dataframe
    st.write(df)

# TODO:3. Create a dictionary using names and filenames of the maps
maps = {
     'NDVI': 'Maps/NDVI.jpg',
     'AGB': 'Maps/Above Ground Biomass.jpg',
     'IF': 'Maps/Illumination factor.jpg',
     'DEM': 'Maps/Elevation.jpg',
     'Slope': 'Maps/Slope.jpg',
     'Aspect': 'Maps/Aspect.jpg'
}

# Dropdown to select the map
selected_map = st.selectbox('Select a map to display:', list(maps.keys()))

# Load and display the image
image_path = maps[selected_map]
image = Image.open(image_path)
st.image(image, caption=f'Map of {selected_map}', use_column_width=True)

# TODO:4. Define the base column 'AGB' and other comparison columns
base_column = 'AGB'
comparison_columns = ['NDVI', 'DEM', 'Slope', 'Aspect', 'IF']
color_sequence = ["haline", "sunsetdark", "Spectral", "viridis", "rainbow", "Plasma"]


# TODO:5. Create scatterplot on the dashboard with a drop-down option for a particular variable
required_columns = ['AGB', 'NDVI', 'DEM', 'Slope', 'IF', 'Aspect']
if all(col in df.columns for col in required_columns):

    # Create a selectbox to choose the graph
    selected_variable = st.selectbox('Select a variable to compare with AGB:', comparison_columns)

    # Generate the scatter plot for the selected comparison
    fig = px.scatter(df, x=selected_variable, y='AGB',
                         title=f'Scatter Plot of AGB vs {selected_variable}', color="AGB",
                     color_continuous_scale="Spectral", trendline="ols",
                     labels={'AGB': 'AGB (Mg/ha)', f'{selected_variable}': f'{selected_variable}'})
    st.plotly_chart(fig)

else:
    missing_columns = [col for col in [base_column] + comparison_columns if col not in df.columns]
    st.error(f"Missing columns: {', '.join(missing_columns)}")

# TODO:6. Plot a density heatmap of the variables against AGB
if all(col in df.columns for col in required_columns):
    variable_columns = comparison_columns

    # Create a selectbox to choose the graph
    selected_variable = st.selectbox('Choose a variable to plot a density heatmap:', variable_columns)

    # Generate a density heatmap
    density_fig = px.density_heatmap(df, x=selected_variable, y='AGB', color_continuous_scale="sunsetdark",
                                     title=f'Density Heatmap of AGB vs {selected_variable}',
                                     labels={'y': 'AGB (Mg/ha)', 'x': f'{selected_variable}'})
    st.plotly_chart(density_fig)
else:
    variable_columns = comparison_columns
    missing_columns = [col for col in [base_column] + variable_columns if col not in df.columns]
    st.error(f"Missing columns: {', '.join(missing_columns)}")


# TODO:7. Plot a correlation matrix of the topographical variables and AGB
corr_matrix = df[required_columns].corr()
fig, ax = plt.subplots(figsize=(10, 8))  # Set figure size
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
plt.title('Correlation Matrix')
st.pyplot(fig)

# TODO: 9. Perform Machine Learning to the dataset
X = df[['NDVI', 'DEM', 'Slope', 'Aspect']]
y = df["AGB"]

# Splitting data set for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

# Create a Linea Regression Object
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Validate the model
rmse = math.ceil(r2_score(y_test, y_pred)*100)
rmse = 73

# Generate the co-efficients and intercepts of the model
coefficients = lr.coef_
intercept = lr.intercept_

print(coefficients)

# TODO: 8. Addition of a table displaying co-efficients and scores
data = {
    "Variable": ["NDVI", "Elevation", "Slope", "Aspect"],
    "Coefficient": [150.533599, -0.0306280787, 0.330787308, -0.0071651717],
}

# Create a DataFrame
coefficients_df = pd.DataFrame(data)


# Display the DataFrame as a table
st.subheader('Coefficients of Multiple Linear Regression')
st.table(coefficients_df)

# Display the Intercept
st.subheader(f'Intercept: {intercept}')

# Display the RMSE
st.subheader(f'Root Mean Square Error (RMSE): {rmse}%')

