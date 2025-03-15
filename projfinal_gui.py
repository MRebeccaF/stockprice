import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn as sk
import sklearn.preprocessing as skp
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import sklearn.svm as sks
import xgboost as xg
import warnings
import tkinter as tk
import turtle
import tkinter.font as font

warnings.filterwarnings('ignore')

def is_month_end(row):
    day = int(row['day'])
    month = int(row['month'])
    if (day == 31 and month in [1, 3, 5, 7, 8, 10, 12]) or (day == 30 and month in [4, 6, 9, 11]) or (day == 28 and month == 2):
        return 1
    else:
        return 0

def plotprice():
    plt.figure(figsize=(15,5))
    plt.plot(df['Open'])
    plt.title('Techmahindra Open Price', fontsize=15)
    plt.ylabel('Price in rupees')
    plt.show()

def distplot():
    columns=[(1,'Open'),(2,'High'),(3,'Low'),(4,'Close'),(5,'Volume')]
    plt.subplots(figsize=(20,10))
    for i, col in columns:
        plt.subplot(2,3,i)
        sb.distplot(df[col])
    plt.show()

def boxplot():
    columns=[(1,'Open'),(2,'High'),(3,'Low'),(4,'Close'),(5,'Volume')]
    plt.subplots(figsize=(20,10))
    for i, col in columns:
        plt.subplot(2,3,i)
        sb.boxplot(df[col])
    plt.show()

def bargraph():
    grouped = df.groupby('month').mean(numeric_only=True)
    plt.subplots(figsize=(20,10))
    columns=[(1,'Open'),(2,'High'),(3,'Low'),(4,'Close')]
    for i, col in columns:
        plt.subplot(2,2,i)
        grouped[col].plot.bar()
    plt.show()

def checktargetbalance():
    plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
    plt.show()

def checkcorrheatmap():
    plt.figure(figsize=(10, 10))
    sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
    plt.show()

def split_and_normalise():
    print(X_train.shape, X_valid.shape)
    for i in range(3):
        models[i].fit(X_train, Y_train)

        global last_fitted_model
        last_fitted_model = models[i]
        
        print(f'{models[i]} : ')
        print('Training Accuracy : ', sk.metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
        print('Validation Accuracy : ', sk.metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
        print()

def confusion_matrix():
    for i in range(3):
        models[i].fit(X_train, Y_train)
        predicted = models[i].predict(X_valid)
        cm = sk.metrics.confusion_matrix(Y_valid, predicted)
        cmdisp = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        cmdisp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {models[i]}')
        plt.show()

def plot_future_stock_prices(trained_model, future_data):    
    features = future_data[['open-close', 'low-high', 'is_month_end']]      # Preprocess the features using the same scaler used during training
    scaler = skp.StandardScaler()
    features_scaled = scaler.fit_transform(features)

    predicted_probabilities = trained_model.predict_proba(features_scaled)[:, 1]    # Make predictions using the trained model

    dates = future_data['Date']

    plt.figure(figsize=(10, 7))             # Plotting the predicted probabilities
    plt.plot(dates, predicted_probabilities, label='Predicted Probabilities', marker='o', linestyle='-', color='b')
    plt.title('Predicted Probabilities for Future Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Predicted Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

def dispfutureprice():
    if last_fitted_model is not None:
        plot_future_stock_prices(last_fitted_model, future_data)
        print()
    else:
        tk.messagebox.showinfo( 'Message', 'Please train a model before predicting future stock prices')

def baj():
    global df
    df=pd.read_csv('BAJAJ-AUTO.csv')
    df = df.drop(['Adj Close'], axis=1)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']

    df[['year','month','day']] = df['Date'].str.split('-',expand=True)
    df['day'] = df['day'].astype('int')
    df['month'] = df['month'].astype('int')
    df['year'] = df['year'].astype('int')

    df['is_month_end'] = df.apply(is_month_end, axis=1)
    df = df.drop(['Date'], axis=1)

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = df[['open-close', 'low-high', 'is_month_end']]
    target = df['target']

    scaler = skp.StandardScaler()
    features = scaler.fit_transform(features)

    global X_train
    global X_valid
    global Y_train
    global Y_valid
    X_train, X_valid, Y_train, Y_valid = skm.train_test_split(features, target, test_size=0.2, random_state=2022)

    global models
    models = [skl.LogisticRegression(), sks.SVC(kernel='poly', probability=True), xg.XGBClassifier()]

    global last_fitted_model
    last_fitted_model=None

    global future_data
    future_data = pd.DataFrame({
    'open-close': np.random.rand(100),
    'low-high': np.random.rand(100),
    'is_month_end': np.random.choice([0, 1], size=100),
    'Date': pd.date_range(start='2023-05-05', periods=100)
    })                                                              #Sample dataset whose values will be replaced with predicted values

    mainframe.pack_forget()
    bajfr=tk.Frame()
    myfont = font.Font(size=20)
    heading=tk.Label(bajfr, text ='BAJAJ-AUTO',font=myfont,bg='red',fg='white')
    a=tk.Button(bajfr, text='Print the price distribution of the stock', width=100, font=myfont, bg='purple', fg='white', command=plotprice)
    b=tk.Button(bajfr, text='Print the distplot for the data', width=100, font=myfont, bg='violet', command=distplot)
    c=tk.Button(bajfr, text='Print the boxplot for the data', width=100, font=myfont, bg='purple', fg='white', command=boxplot)
    d=tk.Button(bajfr, text='Print the bar graph for the data', width=100, font=myfont, bg='violet', command=bargraph)
    e=tk.Button(bajfr, text='Check the balance for the target column', width=100, font=myfont, bg='purple', fg='white', command=checktargetbalance)
    f=tk.Button(bajfr, text='Check the correlation heatmap', width=100, font=myfont, bg='violet', command=checkcorrheatmap)
    g=tk.Button(bajfr, text='Train models and show the accuracy of the prediction models', width=100, font=myfont, bg='purple', fg='white', command=split_and_normalise)
    h=tk.Button(bajfr, text='Plot a confusion matrix for the predicted data', width=100, font=myfont, bg='violet', command=confusion_matrix)
    i=tk.Button(bajfr, text='Show future stock price trend', width=100, font=myfont, bg='purple', fg='white', command=dispfutureprice)
    j=tk.Button(bajfr, text='Go to main menu', width=100, font=myfont, bg='violet', command=lambda:[bajfr.pack_forget(), mainframe.pack()])
    heading.pack()
    bajfr.pack()
    a.pack()
    b.pack()
    c.pack()
    d.pack()
    e.pack()
    f.pack()
    g.pack()
    h.pack()
    i.pack()
    j.pack()
    
def eich():
    global df
    df=pd.read_csv('EICHERMOT.csv')
    df = df.drop(['Adj Close'], axis=1)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']

    df[['year','month','day']] = df['Date'].str.split('-',expand=True)
    df['day'] = df['day'].astype('int')
    df['month'] = df['month'].astype('int')
    df['year'] = df['year'].astype('int')

    df['is_month_end'] = df.apply(is_month_end, axis=1)
    df = df.drop(['Date'], axis=1)

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = df[['open-close', 'low-high', 'is_month_end']]
    target = df['target']

    scaler = skp.StandardScaler()
    features = scaler.fit_transform(features)

    global X_train
    global X_valid
    global Y_train
    global Y_valid
    X_train, X_valid, Y_train, Y_valid = skm.train_test_split(features, target, test_size=0.2, random_state=2022)

    global models
    models = [skl.LogisticRegression(), sks.SVC(kernel='poly', probability=True), xg.XGBClassifier()]

    global last_fitted_model
    last_fitted_model=None

    global future_data
    future_data = pd.DataFrame({
    'open-close': np.random.rand(100),
    'low-high': np.random.rand(100),
    'is_month_end': np.random.choice([0, 1], size=100),
    'Date': pd.date_range(start='2023-05-05', periods=100)
    })                                                              #Sample dataset whose values will be replaced with predicted values

    mainframe.pack_forget()
    eichfr=tk.Frame()
    myfont = font.Font(size=20)
    heading=tk.Label(eichfr, text ='EICHERMOT',font=myfont,bg='green',fg='white')
    a=tk.Button(eichfr, text='Print the price distribution of the stock', width=100, font=myfont, bg='purple', fg='white', command=plotprice)
    b=tk.Button(eichfr, text='Print the distplot for the data', width=100, font=myfont, bg='violet', command=distplot)
    c=tk.Button(eichfr, text='Print the boxplot for the data', width=100, font=myfont, bg='purple', fg='white', command=boxplot)
    d=tk.Button(eichfr, text='Print the bar graph for the data', width=100, font=myfont, bg='violet', command=bargraph)
    e=tk.Button(eichfr, text='Check the balance for the target column', width=100, font=myfont, bg='purple', fg='white', command=checktargetbalance)
    f=tk.Button(eichfr, text='Check the correlation heatmap', width=100, font=myfont, bg='violet', command=checkcorrheatmap)
    g=tk.Button(eichfr, text='Train models and show the accuracy of the prediction models', width=100, font=myfont, bg='purple', fg='white', command=split_and_normalise)
    h=tk.Button(eichfr, text='Plot a confusion matrix for the predicted data', width=100, font=myfont, bg='violet', command=confusion_matrix)
    i=tk.Button(eichfr, text='Show future stock price trend', width=100, font=myfont, bg='purple', fg='white', command=dispfutureprice)
    j=tk.Button(eichfr, text='Go to main menu', width=100, font=myfont, bg='violet', command=lambda:[eichfr.pack_forget(), mainframe.pack()])
    heading.pack()
    eichfr.pack()
    a.pack()
    b.pack()
    c.pack()
    d.pack()
    e.pack()
    f.pack()
    g.pack()
    h.pack()
    i.pack()
    j.pack()
    
def mar():
    global df
    df=pd.read_csv('MARUTI.csv')
    df = df.drop(['Adj Close'], axis=1)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']

    df[['year','month','day']] = df['Date'].str.split('-',expand=True)
    df['day'] = df['day'].astype('int')
    df['month'] = df['month'].astype('int')
    df['year'] = df['year'].astype('int')

    df['is_month_end'] = df.apply(is_month_end, axis=1)
    df = df.drop(['Date'], axis=1)

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = df[['open-close', 'low-high', 'is_month_end']]
    target = df['target']

    scaler = skp.StandardScaler()
    features = scaler.fit_transform(features)

    global X_train
    global X_valid
    global Y_train
    global Y_valid
    X_train, X_valid, Y_train, Y_valid = skm.train_test_split(features, target, test_size=0.2, random_state=2022)

    global models
    models = [skl.LogisticRegression(), sks.SVC(kernel='poly', probability=True), xg.XGBClassifier()]

    global last_fitted_model
    last_fitted_model=None

    global future_data
    future_data = pd.DataFrame({
    'open-close': np.random.rand(100),
    'low-high': np.random.rand(100),
    'is_month_end': np.random.choice([0, 1], size=100),
    'Date': pd.date_range(start='2023-05-05', periods=100)
    })                                                              #Sample dataset whose values will be replaced with predicted values

    mainframe.pack_forget()
    marfr=tk.Frame()
    myfont = font.Font(size=20)
    heading=tk.Label(marfr, text ='MARUTI',font=myfont,bg='blue',fg='white')
    a=tk.Button(marfr, text='Print the price distribution of the stock', width=100, font=myfont, bg='purple', fg='white', command=plotprice)
    b=tk.Button(marfr, text='Print the distplot for the data', width=100, font=myfont, bg='violet', command=distplot)
    c=tk.Button(marfr, text='Print the boxplot for the data', width=100, font=myfont, bg='purple', fg='white', command=boxplot)
    d=tk.Button(marfr, text='Print the bar graph for the data', width=100, font=myfont, bg='violet', command=bargraph)
    e=tk.Button(marfr, text='Check the balance for the target column', width=100, font=myfont, bg='purple', fg='white', command=checktargetbalance)
    f=tk.Button(marfr, text='Check the correlation heatmap', width=100, font=myfont, bg='violet', command=checkcorrheatmap)
    g=tk.Button(marfr, text='Train models and show the accuracy of the prediction models', width=100, font=myfont, bg='purple', fg='white', command=split_and_normalise)
    h=tk.Button(marfr, text='Plot a confusion matrix for the predicted data', width=100, font=myfont, bg='violet', command=confusion_matrix)
    i=tk.Button(marfr, text='Show future stock price trend', width=100, font=myfont, bg='purple', fg='white', command=dispfutureprice)
    j=tk.Button(marfr, text='Go to main menu', width=100, font=myfont, bg='violet', command=lambda:[marfr.pack_forget(), mainframe.pack()])
    heading.pack()
    marfr.pack()
    a.pack()
    b.pack()
    c.pack()
    d.pack()
    e.pack()
    f.pack()
    g.pack()
    h.pack()
    i.pack()
    j.pack()
    
def tat():
    global df
    df=pd.read_csv('TATAMOTORS.csv')
    df = df.drop(['Adj Close'], axis=1)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']

    df[['year','month','day']] = df['Date'].str.split('-',expand=True)
    df['day'] = df['day'].astype('int')
    df['month'] = df['month'].astype('int')
    df['year'] = df['year'].astype('int')

    df['is_month_end'] = df.apply(is_month_end, axis=1)
    df = df.drop(['Date'], axis=1)

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = df[['open-close', 'low-high', 'is_month_end']]
    target = df['target']

    scaler = skp.StandardScaler()
    features = scaler.fit_transform(features)

    global X_train
    global X_valid
    global Y_train
    global Y_valid
    X_train, X_valid, Y_train, Y_valid = skm.train_test_split(features, target, test_size=0.2, random_state=2022)

    global models
    models = [skl.LogisticRegression(), sks.SVC(kernel='poly', probability=True), xg.XGBClassifier()]

    global last_fitted_model
    last_fitted_model=None

    global future_data
    future_data = pd.DataFrame({
    'open-close': np.random.rand(100),
    'low-high': np.random.rand(100),
    'is_month_end': np.random.choice([0, 1], size=100),
    'Date': pd.date_range(start='2023-05-05', periods=100)
    })                                                              #Sample dataset whose values will be replaced with predicted values

    mainframe.pack_forget()
    tatfr=tk.Frame()
    myfont = font.Font(size=20)
    heading=tk.Label(tatfr, text ='TATAMOTORS',font=myfont,bg='yellow',fg='white')
    a=tk.Button(tatfr, text='Print the price distribution of the stock', width=100, font=myfont, bg='purple', fg='white', command=plotprice)
    b=tk.Button(tatfr, text='Print the distplot for the data', width=100, font=myfont, bg='violet', command=distplot)
    c=tk.Button(tatfr, text='Print the boxplot for the data', width=100, font=myfont, bg='purple', fg='white', command=boxplot)
    d=tk.Button(tatfr, text='Print the bar graph for the data', width=100, font=myfont, bg='violet', command=bargraph)
    e=tk.Button(tatfr, text='Check the balance for the target column', width=100, font=myfont, bg='purple', fg='white', command=checktargetbalance)
    f=tk.Button(tatfr, text='Check the correlation heatmap', width=100, font=myfont, bg='violet', command=checkcorrheatmap)
    g=tk.Button(tatfr, text='Train models and show the accuracy of the prediction models', width=100, font=myfont, bg='purple', fg='white', command=split_and_normalise)
    h=tk.Button(tatfr, text='Plot a confusion matrix for the predicted data', width=100, font=myfont, bg='violet', command=confusion_matrix)
    i=tk.Button(tatfr, text='Show future stock price trend', width=100, font=myfont, bg='purple', fg='white', command=dispfutureprice)
    j=tk.Button(tatfr, text='Go to main menu', width=100, font=myfont, bg='violet', command=lambda:[tatfr.pack_forget(), mainframe.pack()])
    heading.pack()
    tatfr.pack()
    a.pack()
    b.pack()
    c.pack()
    d.pack()
    e.pack()
    f.pack()
    g.pack()
    h.pack()
    i.pack()
    j.pack()
    
def tech():
    global df
    df=pd.read_csv('TECHM.csv')
    df = df.drop(['Adj Close'], axis=1)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']

    df[['year','month','day']] = df['Date'].str.split('-',expand=True)
    df['day'] = df['day'].astype('int')
    df['month'] = df['month'].astype('int')
    df['year'] = df['year'].astype('int')

    df['is_month_end'] = df.apply(is_month_end, axis=1)
    df = df.drop(['Date'], axis=1)

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = df[['open-close', 'low-high', 'is_month_end']]
    target = df['target']

    scaler = skp.StandardScaler()
    features = scaler.fit_transform(features)

    global X_train
    global X_valid
    global Y_train
    global Y_valid
    X_train, X_valid, Y_train, Y_valid = skm.train_test_split(features, target, test_size=0.2, random_state=2022)

    global models
    models = [skl.LogisticRegression(), sks.SVC(kernel='poly', probability=True), xg.XGBClassifier()]

    global last_fitted_model
    last_fitted_model=None

    global future_data
    future_data = pd.DataFrame({
    'open-close': np.random.rand(100),
    'low-high': np.random.rand(100),
    'is_month_end': np.random.choice([0, 1], size=100),
    'Date': pd.date_range(start='2023-05-05', periods=100)
    })                                                              #Sample dataset whose values will be replaced with predicted values

    mainframe.pack_forget()
    techfr=tk.Frame()
    myfont = font.Font(size=20)
    heading=tk.Label(techfr, text ='TECHM',font=myfont,bg='orange',fg='white')
    a=tk.Button(techfr, text='Print the price distribution of the stock', width=100, font=myfont, bg='purple', fg='white', command=plotprice)
    b=tk.Button(techfr, text='Print the distplot for the data', width=100, font=myfont, bg='violet', command=distplot)
    c=tk.Button(techfr, text='Print the boxplot for the data', width=100, font=myfont, bg='purple', fg='white', command=boxplot)
    d=tk.Button(techfr, text='Print the bar graph for the data', width=100, font=myfont, bg='violet', command=bargraph)
    e=tk.Button(techfr, text='Check the balance for the target column', width=100, font=myfont, bg='purple', fg='white', command=checktargetbalance)
    f=tk.Button(techfr, text='Check the correlation heatmap', width=100, font=myfont, bg='violet', command=checkcorrheatmap)
    g=tk.Button(techfr, text='Train models and show the accuracy of the prediction models', width=100, font=myfont, bg='purple', fg='white', command=split_and_normalise)
    h=tk.Button(techfr, text='Plot a confusion matrix for the predicted data', width=100, font=myfont, bg='violet', command=confusion_matrix)
    i=tk.Button(techfr, text='Show future stock price trend', width=100, font=myfont, bg='purple', fg='white', command=dispfutureprice)
    j=tk.Button(techfr, text='Go to main menu', width=100, font=myfont, bg='violet', command=lambda:[techfr.pack_forget(), mainframe.pack()])
    heading.pack()
    techfr.pack()
    a.pack()
    b.pack()
    c.pack()
    d.pack()
    e.pack()
    f.pack()
    g.pack()
    h.pack()
    i.pack()
    j.pack()

def thankyou():
    turtle.TurtleScreen._RUNNING=True
    scr=turtle.Screen()
    tur=turtle.Turtle()
    tur.color('green')
    tur.write('THANK YOU',move=True, align='center',font=('Arial', 30, 'normal'))
    turtle.exitonclick()

scr=turtle.Screen()
tur=turtle.Turtle()
tur.color('brown')
tur.write('WELCOME TO STOCK PRICE PREDICTION',move=True, align='center',font=('Arial', 25, 'normal'))
turtle.exitonclick()

tkin=tk.Tk()
myfont1=font.Font(size=30)
myfont2=font.Font(size=20)

mainframe=tk.Frame()
stock=tk.Label(mainframe, text ='SELECT ANY STOCK',font=myfont1,bg='light green')
bajaj = tk.Button(mainframe, text='BAJAJ-AUTO', width=50, font=myfont1, bg='red', command=baj)
eichermot = tk.Button(mainframe, text='EICHERMOT', width=50, font=myfont1, bg='green', command=eich)
maruti = tk.Button(mainframe, text='MARUTI', width=50, font=myfont1, bg='blue', command=mar)
tatmot = tk.Button(mainframe, text='TATAMOTORS', width=50, font=myfont1, bg='yellow', command=tat)
techm = tk.Button(mainframe, text='TECHM', width=50, font=myfont1, bg='orange', command=tech)
Exit = tk.Button(mainframe, text='EXIT', width=25, font=myfont2, bg='purple', fg='white', command=tkin.destroy)
mainframe.pack()
stock.pack()
bajaj.pack()
eichermot.pack()
maruti.pack()
tatmot.pack()
techm.pack()
Exit.pack()
tkin.mainloop()

thankyou()
