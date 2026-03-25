from django.db.models import Count, Avg, Q
from django.shortcuts import render, redirect
import datetime
import xlwt
from django.http import HttpResponse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from Remote_User.models import (
    ClientRegister_Model,
    detect_cyber_threat,
    detection_ratio,
    detection_accuracy
)


# ---------------------------------
# Service Provider Login
# ---------------------------------
def serviceproviderlogin(request):
    if request.method == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')

        if admin == "Admin" and password == "Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request, 'SProvider/serviceproviderlogin.html')


# ---------------------------------
# View Cyber Threat Ratio
# ---------------------------------
def View_Cyber_Threat_Type_Ratio(request):
    detection_ratio.objects.all().delete()

    total_count = detect_cyber_threat.objects.count()

    if total_count == 0:
        return render(request, 'SProvider/View_Cyber_Threat_Type_Ratio.html', {'objs': []})

    # Packet Drop
    kword = 'Packet Drop'
    count = detect_cyber_threat.objects.filter(Prediction=kword).count()
    ratio = (count / total_count) * 100

    if ratio > 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    # Packet Hijacking
    kword12 = 'Packet Hijacking'
    count12 = detect_cyber_threat.objects.filter(Prediction=kword12).count()
    ratio12 = (count12 / total_count) * 100

    if ratio12 > 0:
        detection_ratio.objects.create(names=kword12, ratio=ratio12)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Cyber_Threat_Type_Ratio.html', {'objs': obj})


# ---------------------------------
# View Remote Users
# ---------------------------------
def View_Remote_Users(request):
    obj = ClientRegister_Model.objects.all()
    return render(request, 'SProvider/View_Remote_Users.html', {'objects': obj})


# ---------------------------------
# View Trending Topics
# ---------------------------------
def ViewTrendings(request):
    topic = detect_cyber_threat.objects.values('topics') \
        .annotate(dcount=Count('topics')) \
        .order_by('-dcount')

    return render(request, 'SProvider/ViewTrendings.html', {'objects': topic})


# ---------------------------------
# Charts - Ratio
# ---------------------------------
def charts(request, chart_type):
    chart1 = detection_ratio.objects.values('names') \
        .annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts.html", {'form': chart1, 'chart_type': chart_type})


# ---------------------------------
# Charts - Accuracy
# ---------------------------------
def charts1(request, chart_type):
    chart1 = detection_accuracy.objects.values('names') \
        .annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts1.html", {'form': chart1, 'chart_type': chart_type})


# ---------------------------------
# View Predictions
# ---------------------------------
def View_Prediction_Of_Cyber_Threat_Type(request):
    obj = detect_cyber_threat.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Cyber_Threat_Type.html', {'list_objects': obj})


# ---------------------------------
# Like Chart
# ---------------------------------
def likeschart(request, like_chart):
    charts = detection_accuracy.objects.values('names') \
        .annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/likeschart.html", {'form': charts, 'like_chart': like_chart})


# ---------------------------------
# Download Dataset
# ---------------------------------
def Download_Predicted_DataSets(request):
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Predicted_Data.xls"'

    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet("sheet1")

    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True

    columns = [
        'PID', 'Time', 'Source IP', 'Destination IP',
        'Frame Protocols', 'Source Port', 'Destination Port',
        'Bytes Transferred', 'Protocol', 'Date', 'Prediction'
    ]

    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)

    font_style.font.bold = False

    obj = detect_cyber_threat.objects.all()

    for my_row in obj:
        row_num += 1
        ws.write(row_num, 0, my_row.pid)
        ws.write(row_num, 1, my_row.ptime)
        ws.write(row_num, 2, my_row.src_ip_address)
        ws.write(row_num, 3, my_row.dst_ip_address)
        ws.write(row_num, 4, my_row.frame_protos)
        ws.write(row_num, 5, my_row.src_port)
        ws.write(row_num, 6, my_row.dst_port)
        ws.write(row_num, 7, my_row.bytes_trans)
        ws.write(row_num, 8, my_row.protocol)
        ws.write(row_num, 9, my_row.Date1)
        ws.write(row_num, 10, my_row.Prediction)

    wb.save(response)
    return response


# ---------------------------------
# Train Model (FIXED & IMPROVED)
# ---------------------------------
def train_model(request):

    detection_accuracy.objects.all().delete()

    dataset = pd.read_csv("IIoT_Network_Datasets.csv", encoding='latin-1')

    def apply_results(label):
        if label == 0:
            return 0
        elif label == 1:
            return 1
        return 0

    dataset['Results'] = dataset['attack'].apply(apply_results)

    # Convert protocol to string and one-hot encode
    dataset['protocol'] = dataset['protocol'].astype(str)
    dataset = pd.get_dummies(dataset, columns=['protocol'], drop_first=True)

    #X = dataset.drop(['Results', 'pid'], axis=1)
    X = dataset.drop([
    'pid',
    'ptime',
    'src_ip_address',
    'dst_ip_address',
    'frame_protos',
    'Date',
    'Results'
    ], axis=1)
    
    y = dataset['Results']

    # -------- SCALING --------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------- BALANCING DATA (IMPORTANT FIX) --------
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # -------- TRAIN TEST SPLIT --------
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.20,
        random_state=42
    )

    # -------- Logistic Regression --------
    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(
        solver='lbfgs',
        max_iter=10000
    )

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    log_acc = accuracy_score(y_test, y_pred) * 100

    detection_accuracy.objects.create(
        names="Logistic Regression",
        ratio=log_acc
    )

    # -------- Random Forest (Better Model) --------
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred) * 100

    detection_accuracy.objects.create(
        names="Random Forest",
        ratio=rf_acc
    )

    dataset.to_csv('Labled_data.csv', index=False)

    obj = detection_accuracy.objects.all()
    return render(request, 'SProvider/train_model.html', {'objs': obj})