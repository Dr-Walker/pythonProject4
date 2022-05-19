
import matplotlib.pyplot
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.special import lpmv as lpmv
from math import factorial as factorial
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 711)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.titlelabel = QtWidgets.QLabel(self.centralwidget)
        self.titlelabel.setGeometry(QtCore.QRect(20, 10, 671, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.titlelabel.setFont(font)
        self.titlelabel.setObjectName("titlelabel")
        self.nlabel = QtWidgets.QLabel(self.centralwidget)
        self.nlabel.setGeometry(QtCore.QRect(6, 81, 91, 29))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.nlabel.setFont(font)
        self.nlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.nlabel.setObjectName("nlabel")
        self.llabel = QtWidgets.QLabel(self.centralwidget)
        self.llabel.setGeometry(QtCore.QRect(8, 116, 81, 29))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.llabel.setFont(font)
        self.llabel.setAlignment(QtCore.Qt.AlignCenter)
        self.llabel.setObjectName("llabel")
        self.mlabel = QtWidgets.QLabel(self.centralwidget)
        self.mlabel.setGeometry(QtCore.QRect(13, 151, 91, 29))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.mlabel.setFont(font)
        self.mlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.mlabel.setObjectName("mlabel")
        self.nBox = QtWidgets.QComboBox(self.centralwidget)
        self.nBox.setGeometry(QtCore.QRect(110, 80, 60, 22))
        self.nBox.setObjectName("nBox")
        self.nBox.addItem("")
        self.nBox.addItem("")
        self.nBox.addItem("")
        self.nBox.addItem("")
        self.nBox.addItem("")
        self.nBox.addItem("")
        self.lBox = QtWidgets.QComboBox(self.centralwidget)
        self.lBox.setGeometry(QtCore.QRect(110, 120, 60, 22))
        self.lBox.setObjectName("lBox")
        self.lBox.addItem("")
        self.lBox.addItem("")
        self.lBox.addItem("")
        self.lBox.addItem("")
        self.lBox.addItem("")
        self.lBox.addItem("")
        self.mBox = QtWidgets.QComboBox(self.centralwidget)
        self.mBox.setGeometry(QtCore.QRect(110, 160, 60, 22))
        self.mBox.setObjectName("mBox")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.mBox.addItem("")
        self.RadialBut = QtWidgets.QPushButton(self.centralwidget)
        self.RadialBut.setGeometry(QtCore.QRect(60, 260, 181, 141))
        self.RadialBut.setObjectName("RadialBut")
        self.Orbbutton = QtWidgets.QPushButton(self.centralwidget)
        self.Orbbutton.setGeometry(QtCore.QRect(310, 260, 181, 141))
        self.Orbbutton.setObjectName("Orbbutton")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(460, 80, 261, 141))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame.setFont(font)
        self.frame.setStyleSheet("background-color: rgb(255, 61, 2);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")


        self.Commentlab = QtWidgets.QLabel(self.frame)
        self.Commentlab.setGeometry(QtCore.QRect(20, 20, 201, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Commentlab.setFont(font)
        self.Commentlab.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.Commentlab.setWordWrap(True)
        self.Commentlab.setObjectName("Commentlab")



        self.Cboxlabel = QtWidgets.QLabel(self.centralwidget)
        self.Cboxlabel.setGeometry(QtCore.QRect(470, 50, 171, 21))
        self.Cboxlabel.setObjectName("Cboxlabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.RadialBut.clicked.connect(self.Radclicked)
        self.Orbbutton.clicked.connect(self.pltorbclicked)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.titlelabel.setText(_translate("MainWindow", "Which orbital are you interested in?"))
        self.nlabel.setText(_translate("MainWindow", "n = "))
        self.llabel.setText(_translate("MainWindow", "l = "))
        self.mlabel.setText(_translate("MainWindow", "m = "))
        self.nBox.setItemText(0, _translate("MainWindow", "1"))
        self.nBox.setItemText(1, _translate("MainWindow", "2"))
        self.nBox.setItemText(2, _translate("MainWindow", "3"))
        self.nBox.setItemText(3, _translate("MainWindow", "4"))
        self.nBox.setItemText(4, _translate("MainWindow", "5"))
        self.nBox.setItemText(5, _translate("MainWindow", "6"))
        self.lBox.setItemText(0, _translate("MainWindow", "0"))
        self.lBox.setItemText(1, _translate("MainWindow", "1"))
        self.lBox.setItemText(2, _translate("MainWindow", "2"))
        self.lBox.setItemText(3, _translate("MainWindow", "3"))
        self.lBox.setItemText(4, _translate("MainWindow", "4"))
        self.lBox.setItemText(5, _translate("MainWindow", "5"))
        self.mBox.setItemText(0, _translate("MainWindow", "-5"))
        self.mBox.setItemText(1, _translate("MainWindow", "-4"))
        self.mBox.setItemText(2, _translate("MainWindow", "-3"))
        self.mBox.setItemText(3, _translate("MainWindow", "-2"))
        self.mBox.setItemText(4, _translate("MainWindow", "-1"))
        self.mBox.setItemText(5, _translate("MainWindow", "0"))
        self.mBox.setItemText(6, _translate("MainWindow", "1"))
        self.mBox.setItemText(7, _translate("MainWindow", "2"))
        self.mBox.setItemText(8, _translate("MainWindow", "3"))
        self.mBox.setItemText(9, _translate("MainWindow", "4"))
        self.mBox.setItemText(10, _translate("MainWindow", "5"))
        self.RadialBut.setText(_translate("MainWindow", "Plot Radial Part"))
        self.Orbbutton.setText(_translate("MainWindow", "Plot Orbital"))
        self.frame.setStatusTip(_translate("MainWindow", "Any errors will show up here"))
        self.Commentlab.setText(_translate("MainWindow", "No Errors"))
        self.Cboxlabel.setText(_translate("MainWindow", "Comment Box"))

    def Radclicked(self):
        n = int(self.nBox.currentText())
        l = int(self.lBox.currentText())
        m = int(self.mBox.currentText())
     #   m = int(self.mBox.currentText())


        if abs(m)> l:
            print("Please check your n,l,m values remember the quantum # rules")
            self.Commentlab.setText("Please check your n,l,m values remember the quantum # rules")
        elif l >= n:
            print("Please check your n,l,m values remember the quantum # rules")
            self.Commentlab.setText("Please check your n,l,m values remember the quantum # rules")
        else:
            self.Commentlab.setText("Processing, check behind the main window for plots.")

            def Plot_R(n, l):
                style.use('fivethirtyeight')
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                newrange = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
                r = np.linspace(0, newrange, 100)

                plt.ylabel("electron density")
                plt.xlabel("r / $a_o$")
                plt.plot(r, psi_R(n, l, r, ))
                plt.tight_layout()
                plt.show()

            def psi_R(n, l, r):
                psi_R = r ** 2 * R(n, l, r) ** 2
                return psi_R

            def AssociatedLegendre(l, m, x):
                lpmv(m, l, x)  # Annoyingly lpmv needs m to come first in the call order.

                return lpmv(m, l, x)

            def AssociatedLaguerre(n, m, x):
                Anm = 0

                for a in range(0, n + 1):
                    Zeta = (factorial(m + n)) / ((factorial((m + n) - (n - a))) * factorial(n - a))
                    Anm = Anm + factorial(n + m) * (Zeta) / factorial(a) * (-x) ** a
                return Anm

            # Spherical Ylm function uses the Associated Legendre function
            def SphericalYlm(l, m, theta, phi):
                SphericalYlm = (-1 ** m) * np.sqrt(
                    ((2 * l + 1) / (4 * np.pi)) * ((factorial(l - abs(m))) / (factorial(l + abs(m))))) \
                               * AssociatedLegendre(l, m, np.cos(theta)) * np.exp(1j * m * phi)
                return SphericalYlm

            def Y(l, m, theta, phi):
                if m < 0:
                    Y = np.sqrt(2) * (-1) ** m * np.imag(SphericalYlm(l, abs(m), theta, phi))
                elif m == 0:
                    Y = (-1 ** m) * np.sqrt(
                        ((2 * l + 1) / (4 * np.pi)) * ((factorial(1 - abs(m))) / (factorial(1 + abs(m))))) \
                        * AssociatedLegendre(l, m, np.cos(theta))
                else:
                    Y = np.sqrt(2) * (-1) ** m * np.real(SphericalYlm(l, abs(m), theta, phi))

                return Y

            def R(n, l, r):
                a = 1
                R = (np.sqrt((2 / (a * n)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) * np.exp(
                    -r / (a * n)) * (2 * r / (a * n)) ** l \
                     / factorial(n - l - 1 + 2 * l + 1) * AssociatedLaguerre(n - l - 1, 2 * l + 1, 2 * r / (a * n)))
                return R
            Plot_R(n,l)
    def pltorbclicked(self):
        n = int(self.nBox.currentText())
        l = int(self.lBox.currentText())
        m = int(self.mBox.currentText())
        #   m = int(self.mBox.currentText())

        if abs(m) > l:

            self.Commentlab.setText("Please check your n,l,m values remember the quantum # rules")
        elif l >= n:

            self.Commentlab.setText("Please check your n,l,m values remember the quantum # rules")
        else:
            self.Commentlab.setText("Processing, check behind the main window for plots.")
            def Plot_psi(n, l, m, ):

                border = 100
                accuracy = 220  # Higher number smoother surface, but longer processign time.
                raster = np.linspace(-border, border, accuracy)
                [x, y, z] = np.meshgrid(raster, raster, raster)
                r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)
                Wave3 = psi_2(n, l, m, r, theta, phi)
                mlab.figure(1, fgcolor=(1, 1, 1))
                # We create a scalar field with the module of Phi as the scalar
                src = mlab.pipeline.scalar_field(Wave3)

                src.image_data.point_data.add_array(np.sign(psi(n, l, m, r, theta, phi)).T.ravel())

                src.image_data.point_data.get_array(1).name = 'phase'
                # Make sure that the dataset is up to date with the different arrays:
                src.update()

                # We select the 'scalar' attribute, ie the norm of Phi
                src2 = mlab.pipeline.set_active_attribute(src,
                                                          point_scalars='scalar')

                # Cut isosurfaces of the norm
                contour = mlab.pipeline.contour(src2)
                contour.filter.contours = [0.0015, ]

                # Now we select the 'angle' attribute, ie the phase of Phi
                contour2 = mlab.pipeline.set_active_attribute(contour,
                                                              point_scalars='phase')

                # And we display the surface. The colormap is the current attribute: the phase.
                mlab.pipeline.surface(contour2, colormap='plasma', opacity=0.5)

                mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)

                mlab.show()

            # Spherical Harmonic functions
            def AssociatedLegendre(l, m, x):
                lpmv(m, l, x)  # Annoyingly lpmv needs m to come first in the call order.

                return lpmv(m, l, x)

            def AssociatedLaguerre(n, m, x):
                Anm = 0

                for a in range(0, n + 1):
                    Zeta = (factorial(m + n)) / ((factorial((m + n) - (n - a))) * factorial(n - a))
                    Anm = Anm + factorial(n + m) * (Zeta) / factorial(a) * (-x) ** a
                return Anm

            # Spherical Ylm function uses the Associated Legendre function
            def SphericalYlm(l, m, theta, phi):
                SphericalYlm = (-1 ** m) * np.sqrt(
                    ((2 * l + 1) / (4 * np.pi)) * ((factorial(l - abs(m))) / (factorial(l + abs(m))))) \
                               * AssociatedLegendre(l, m, np.cos(theta)) * np.exp(1j * m * phi)
                return SphericalYlm

            def Y(l, m, theta, phi):
                if m < 0:
                    Y = np.sqrt(2) * (-1) ** m * np.imag(SphericalYlm(l, abs(m), theta, phi))
                elif m == 0:
                    Y = (-1 ** m) * np.sqrt(
                        ((2 * l + 1) / (4 * np.pi)) * ((factorial(1 - abs(m))) / (factorial(1 + abs(m))))) \
                        * AssociatedLegendre(l, m, np.cos(theta))
                else:
                    Y = np.sqrt(2) * (-1) ** m * np.real(SphericalYlm(l, abs(m), theta, phi))

                return Y

            def R(n, l, r):
                a = 1
                R = (np.sqrt((2 / (a * n)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) * np.exp(
                    -r / (a * n)) * (2 * r / (a * n)) ** l \
                     / factorial(n - l - 1 + 2 * l + 1) * AssociatedLaguerre(n - l - 1, 2 * l + 1, 2 * r / (a * n)))
                return R

            # Wavefunction
            def psi(n, l, m, r, theta, phi):
                psi = R(n, l, r) * Y(l, m, theta, phi)
                return psi

            def psi_R(n, l, r):
                psi_R = r ** 2 * R(n, l, r) ** 2
                return psi_R

            def psi_2(n, l, m, r, theta, phi):
                psi_2 = (r * R(n, l, r) * Y(l, m, theta, phi)) ** 2
                return psi_2

            def Plot_psi2plot1(n, l, m):
                size = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
                X2 = np.linspace(-size, size, 50)
                Y2 = np.linspace(-size, size, 50)
                [X2, Y2] = np.meshgrid(X2, Y2)
                rad = np.sqrt(X2 ** 2 + Y2 ** 2)
                theta = 90
                phi = np.arctan2(Y2, X2)
                Z = psi_2(n, l, m, rad, theta, phi)
                fig, ax = plt.subplots(1, 1)
                cp = ax.contourf(X2, Y2, Z)
                fig.colorbar(cp, label='Probability density')  # Add a colorbar to a plot
                ax.set_title('Filled Contours Plot')
                ax.set_xlabel('r / $a_o$')
                ax.set_ylabel('r / $a_o$')
                plt.show()

            def Plot_psi2plot2(n, l, m):
                size = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
                Y2 = np.linspace(-size, size, 50)
                Z2 = np.linspace(-size, size, 50)
                [Y2, Z2] = np.meshgrid(Y2, Z2)
                rad = np.sqrt(Y2 ** 2 + Z2 ** 2)
                theta = np.arccos(Z2 / rad)
                phi = 90
                X2 = psi_2(n, l, m, rad, theta, phi)
                fig, ax = plt.subplots(1, 1)
                cp = ax.contourf(Y2, Z2, X2)
                fig.colorbar(cp, label='Probability density')  # Add a colorbar to a plot
                ax.set_title('Filled Contours Plot')
                ax.set_xlabel('r / $a_o$')
                ax.set_ylabel('r / $a_o$')
                plt.show()

            def Plot_psi2plot3(n, l, m):
                size = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
                X2 = np.linspace(-size, size, 50)
                Y2 = np.linspace(-size, size, 50)
                [X2, Y2] = np.meshgrid(X2, Y2)
                rad = np.sqrt(X2 ** 2 + Y2 ** 2)
                theta = 90
                phi = np.arctan2(Y2, X2)
                Z = psi_2(n, l, m, rad, theta, phi)
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot_surface(X2, Y2, Z, cmap='viridis', edgecolor='none')
                ax.set_title('Surface plot')
                ax.set_zlabel('probability density')
                ax.set_xlabel('r / $a_o$')
                ax.set_ylabel('r / $a_o$')
                plt.show()

            def intensity_func(n, l, m, r, theta, phi):

                return (psi_2(n, l, m, abs(r), theta, phi))

            def Plot_psi2plot4(n, l, m):
                size = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
                Y2 = np.linspace(-size, size, 50)
                Z2 = np.linspace(-size, size, 50)
                [Y2, Z2] = np.meshgrid(Y2, Z2)
                rad = np.sqrt(Y2 ** 2 + Z2 ** 2)
                theta = np.arccos(Z2 / rad)
                phi = 90
                X2 = psi_2(n, l, m, rad, theta, phi)
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.set_xlabel('r / $a_o$')
                ax.set_ylabel('r / $a_o$')
                ax.set_zlabel('probability density')
                ax.plot_surface(Y2, Z2, X2, cmap='viridis', edgecolor='none')
                ax.set_title('Surface plot')
                plt.show()

            def Ppsi(n, l, m):
                probabilitydensity = 1e-6
                size = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
                border = 80
                accuracy = 150
                raster = np.linspace(-border, border, accuracy)
                [x, y, z] = np.meshgrid(raster, raster, raster)
                r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)
                Wave = psi_2(n, l, m, r, theta, phi).real
                return Wave

            Plot_psi(n,l,m)
            Plot_psi2plot1(n,l,m)
            Plot_psi2plot2(n,l,m)
            Plot_psi2plot3(n,l,m)
            Plot_psi2plot4(n,l,m)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    time.sleep(14)
