from PyQt5.QtWidgets import QMessageBox

'''
def showInfo():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)

    msg.setText("This is a message box")
    msg.setInformativeText("This is additional information")
    msg.setWindowTitle("MessageBox demo")
    msg.setDetailedText("The details are as follows:")
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.buttonClicked.connect(msgbtn)

    retval = msg.exec_()
    print "value of pressed message box button:", retval
'''

def showCritical(message, title="Error de ejecución", details=''):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)

    msg.setText(message)

    msg.setWindowTitle(title)
    if details != '':
        msg.setInformativeText(details)
        msg.setDetailedText("Información adicional:")
    msg.setStandardButtons(QMessageBox.Ok)

    retval = msg.exec_()
    print(f"value of pressed message box button: {retval}")

def showWarning(message, title="Advertencia"):
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Warning)

    msgBox.setText(message)
    msgBox.setWindowTitle(title)
    msgBox.setStandardButtons(QMessageBox.Ok)

    retval = msgBox.exec_()
    print(f"value of pressed message box button: {retval}")