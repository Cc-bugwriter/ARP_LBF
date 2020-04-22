# ARP_LBF
Code fuer ARP
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents
- [Background](#background)
- [Install](#install)
- [Usage](#usage)

## Background
### Zustandsüberwachung von strukturdynamischen Systemen mithilfe von KI-Algorithmen

Aufgrund von externen und internen Anregungen werden mechanische Systeme in Schwingung versetzt. Das Schwingungsverhalten des Systems kann sich durch verschiedene Ursachen, wie z.B. Verschleiß, Lösen von Verbindungen oder Temperatureinflüsse verändern. Anhand der Auswertung von Schwingungsmessdaten können die Ursachen identifiziert und lokalisiert werden. Dies erfolgt in der Regel über eine statistische Auswertung der Messdaten und einem Abgleich mit einem Simulationsmodell. In dieser Arbeit sollen Methoden zur Zustandsüberwachung von mechanischen Systemen anhand von Schwingungsdaten (Eigenfrequenzen, Eigenmoden und Dämpfung) untersucht und anschließend an einem Modell eines Dreimassenschwingers angewendet werden. Das Ziel ist eine automatisierte Auswertung der Daten und ein anschließender Abgleich mit dem zugrundeliegenden Modell. In vorangegangenen Arbeiten wurde ein physikalisches FE-Modell zum Abgleich verwendet. Mit diesem Modell ist es jedoch schwierig nichtlineare Zusammenhänge wie z.B. das Dämpfungsverhalten abzubilden. Dazu soll in dieser Arbeit eine KI eingesetzt werden.

Die Studierenden werden folgende Aufgaben bearbeiten:
- umfassende Literaturrecherche Literaturrecherche zu Zustandsüberwachung mithilfe von KI-Algorithmen
- Aufbau eines KI-Modells des Demonstrators auf Basis eines künstlichen neuronalen Netzes
- Implementierung einer Zustandsüberwachung anhand von Schwingungsdaten
- Validierung der Methodik am Model
- Dokumentation und Präsentation der Ergebnisse

Die Studierenden werden vom Fachgebiet Systemzuverlässigkeit, Adaptronik und Maschinenakustik SAM in Kooperation mit dem Fraunhofer-Institut für Betriebsfestigkeit und Systemzuverlässikgeit LBF betreut. Ihnen werden die notwendigen Arbeitsmittel zur Verfügung gestellt (Rechner, Software, Literatur, Messtechnik etc.). Das Advanced Research Project entspricht den Rahmenbedingungen der geltenden Prüfungsordnung

## Install

This module depends upon a knowledge `Matlab` and `python` 

### Matlab
Um Nutzungseinschränkungen aufgrund der derzeitigen Lage in Bezug auf das Coronavirus zu vermeiden, bietet der Hersteller MathWorks der TU Darmstadt ab sofort bis zum 30.09.2020 eine campusweite Lizenz für MATLAB, Simulink und alle begleitenden Toolboxen sowie einer Reihe von E-Learning-Materialien an.

Studierende, Lehrkräfte und Beschäftigte können unter Verwendung ihrer Universitätsemailadresse einen MathWorks-Account erstellen und individuellen Zugang zu den Installationsmedien über das folgende Portal erhalten:

[MATLAB Lizenz](https://www.mathworks.com/academia/tah-portal/tu-darmstadt-31483887.html)

Tipp: Um Simulation und Maschinen-Learning Toolbox erfolgreich durchzulaufen, soll die Version von MATLAB mindest `2019b` oder `höher` sein. 

### Python
KNN Network wird auf Python Umgebung implementiert, deswegen eine IDE zu installisieren sehr hilfreich ist. Anbei zeigt ein List, deren Applicationen herunterladen und installisieren noetig sind.
- [Python 3.7.7](https://www.python.org/downloads/release/python-377/)
- [Anaconda](https://www.anaconda.com/distribution/)
- [Pycharm](https://www.jetbrains.com/pycharm/download/#section=windows)

## Usage
