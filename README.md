
### Spam Detection Model Using Logistic Regression

This project aims to create a spam detection model that classifies emails as spam or ham (not spam) using a dataset of emails. The process involves data extraction, preprocessing, and applying a Logistic Regression model for binary classification. Below is a step-by-step explanation of the code.

#### Data Extraction and Preprocessing

1. **Importing the Data:**
   ```python
   mail_data = pd.read_csv("/mail_data.csv")
   print(mail_data.shape)
   mail_data.head()
   ```
   - Load the email data from a CSV file and display the shape and first few rows of the dataset.

2. **Analyzing the Distribution:**
   ```python
   mail_data["Category"].value_counts()
   ```
   - Check the distribution of the 'Category' column to see the imbalance between spam and ham emails.

3. **Balancing the Dataset:**
   ```python
   spam = mail_data[mail_data.Category == "spam"]
   ham = mail_data[mail_data.Category == "ham"]
   print(spam.shape)
   print(ham.shape)

   ham = ham.sample(n=750)
   print(ham.shape)
   print(spam.shape)
   ```
   - Separate the spam and ham emails and balance the dataset by sampling the ham emails to match the number of spam emails.

4. **Combining the Data:**
   ```python
   new_mail_data = pd.concat([spam, ham], axis=0)
   print(new_mail_data.shape)
   print(new_mail_data.head())
   ```
   - Combine the balanced spam and ham dataframes into a single dataframe.

5. **Encoding the Category Labels:**
   ```python
   encoder = LabelEncoder()
   new_mail_data["Category"] = encoder.fit_transform(new_mail_data["Category"])
   new_mail_data["Category"].value_counts()
   ```
   - Encode the 'Category' column into binary labels: spam -> 1, ham -> 0.

6. **Text Vectorization:**
   ```python
   x = new_mail_data["Message"]
   y = new_mail_data["Category"]
   extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
   x = extraction.fit_transform(x)
   print(x)
   print(y)
   ```
   - Convert the textual data in the email body to numeric values using TF-IDF vectorization, which considers the importance of words in the context of the entire document.

#### Model Training and Evaluation

1. **Splitting the Data:**
   ```python
   x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.2, stratify=y)
   print(x.shape, x_train.shape, x_test.shape)
   ```
   - Split the data into training and testing sets, ensuring the distribution of labels remains consistent.

2. **Training the Model:**
   ```python
   model = LogisticRegression()
   model.fit(x_train, y_train)
   ```
   - Train a Logistic Regression model on the training data.

3. **Evaluating the Model:**
   ```python
   train_predict = model.predict(x_train)
   train_accuracy = accuracy_score(train_predict, y_train)
   print(train_accuracy)

   test_predict = model.predict(x_test)
   test_accuracy = accuracy_score(test_predict, y_test)
   print(test_accuracy)
   ```
   - Predict and evaluate the model on both training and testing data to ensure no underfitting or overfitting.

#### Prediction Function

1. **Predicting Email Type:**
   ```python
   def mail_predict(input):
       input = extraction.transform(input)
       output = model.predict(input)
       if output == 0:
           print("Ham Mail")
       else:
           print("Spam Mail")

   inp1 = ["WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
   inp2 = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]
   mail_predict(inp1)
   mail_predict(inp2)
   ```
   - Define a function to predict whether an email is spam or ham and test it with example inputs.

This project demonstrates the process of balancing an imbalanced dataset, transforming text data into a numeric format, and applying a Logistic Regression model for classification tasks.
