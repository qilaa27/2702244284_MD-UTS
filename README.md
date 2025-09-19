# Hotel Reservation Status Classifier

This project aims to build and deploy a machine learning model to **predict hotel reservation status**—whether a booking will be *Canceled* or *Not Canceled*—based on the available reservation data.

The model was developed through a complete pipeline of **data preprocessing**, training with a comparison of **Random Forest** and **XGBoost** algorithms, selecting the best-performing model, and saving it using **pickle**.  
A web application was then built with **Streamlit** to allow users to test reservation details interactively.

**Input:** reservation details such as number of adults/children, length of stay, meal plan, room price, and other parameters from the CSV file.  
**Output:** predicted reservation status:

- **Test Case 1 → Not Canceled**  
  ![Test Case 1](Test%20Case%201.png)

- **Test Case 2 → Canceled**  
  ![Test Case 2](Test%20Case%202.png)
