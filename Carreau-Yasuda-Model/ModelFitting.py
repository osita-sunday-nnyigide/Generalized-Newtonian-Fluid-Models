#!/usr/bin/env python

__doc__ = """

This program requires python 3.6 or higher.

This module has the classes and member function(s) used to 

fit data to the Carreau-Yasuda Model.

"""

__author__     = "Osita Sunday Nnyigide"

__copyright__  = "Copyright 2022, Osita Sunday Nnyigide"

__credits__    = ["Hyun Kyu"]

__license__    = "MIT"

__version__    = "1.0.0"

__maintainer__ = "Osita Sunday Nnyigide"

__email__      = "osita@protein-science.com"

__status__     = "Trial"

__date__       = "November 23, 2022"

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import *
from scipy import optimize
from tkinter import filedialog
from matplotlib import gridspec
import matplotlib.pyplot as plt

class GeneralizedNeutonianFluidModels:

    def CarreauYasudaModel(self, x , eta_0, eta_inf, lbda, a, n):
        """
        this function will return the estimated values of viscosity
        with given input data using initial fitting parameters
        """
        return eta_inf + (eta_0 - eta_inf) * (1 + (lbda * x) ** a) ** ((n - 1) / a)

    def FitModel(self, x,y,μo=3354.07,μf=42.2583,λ=2.68884e-5,a=0.902192,n=-1945.61):
        """this function will fit the returned values to experimental data to obtain 
           the optimized values of the fitting parameters of the given model
        """
        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x, y, "ro")
        popt, pcov = optimize.curve_fit(self.CarreauYasudaModel, x, y, p0=[μo , μf, λ, a, n])
        ax1.plot(x, self.CarreauYasudaModel(x, *popt), '--', color ='y', label ="optimized data")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear rate 1/s",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity Pa.s",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)
        plt.text(min(x),min(y)*2,'μo = %8.3f\nμf = %8.3f\nλ = %8.3f\na = %8.3f\nn = %8.3f'%(popt[0],popt[1],popt[2],popt[3],popt[4]))
        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

class GraphicalUserInterface(GeneralizedNeutonianFluidModels):
    """this function starts the initialization of graphical user interface"""
    def __init__(self):

        self.shear_rate=[] 
        self.viscosity=[]
        self.GUI = tk.Tk()
        self.GUI.geometry('1920x1080')
        self.GUI.title('Complex Fluid Laboratory') # give it a title
        self.frame = tk.Frame(self.GUI, height=1080, width=1920, bg='blue')
        self.frame.pack(fill=tk.BOTH, expand=True)
        for i in range(20):
            self.frame.columnconfigure(i, weight=1)
        self.frame.rowconfigure(0, weight=0)

        self.data_x_axis = tk.Text(self.frame, width=20, bg='#F5D0C7', height=20, wrap=WORD)
        self.data_y_axis = tk.Text(self.frame, width=20, bg='#F5D0C7', height=20, wrap=WORD)

        self.zero_vis = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.inf_vis  = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.lamda    = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.trans    = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.pw_indx  = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)

        self.zero_vis_label  = tk.Label(self.frame, text='μo=', width=2, font='none 11', bg='blue', fg='white')
        self.inf_vis_label   = tk.Label(self.frame, text='μf=', width=2, font='none 11', bg='blue', fg='white')
        self.lamda_label     = tk.Label(self.frame, text='λ=',  width=2, font='none 11', bg='blue', fg='white')
        self.trans_label     = tk.Label(self.frame, text='a=',  width=2, font='none 11', bg='blue', fg='white')
        self.pw_indx_label   = tk.Label(self.frame, text='n=',  width=2, font='none 11', bg='blue', fg='white')

        self.upload_label    = tk.Label(self.frame, text='Upload data or paste shear rate and viscosity', bg='blue', fg='white', font='none 12 bold')
        self.upload_txtentry = tk.Entry(self.frame, width=50, bg='white')
        self.upload_txtentry.bind("<Button-1>", lambda e: self.upload_txtentry.delete(0, tk.END))
        self.upload_txtentry.insert(0, "                         *.dat, *.csv or *.txt only")

        self.upload_btn = tk.Button(self.frame, text='Browse', width=7, font='none 12 bold',command=self.osPath, fg='black')
        self.X_Y_label  = tk.Label(self.frame, text='Enter shear rate\nand viscosity\n\nEnter the initial\nFitting parameters',justify=LEFT,width=15,font='none 12',bg='blue',fg='white')

        self.model = StringVar()
        self.model.set("Select Model")
        self.drop_menu= tk.OptionMenu(self.frame, self.model,"CarreauYasuda")
        self.submit_btn = tk.Button(self.frame, text='Submit', width=7, font='none 12 bold', command=self.PlotData)
        self.exit_btn   = tk.Button(self.frame, text='Exit', fg='white', bg='red',width=7, font='none 12 bold', command=self.close_window)

        self.upload_btn.grid(row=2, column=9, padx=30, pady=5, ipady=7, sticky=W)
        self.upload_label.grid(row=1, column=9, padx=45, pady=5)
        self.upload_txtentry.grid(row=2, column=9, padx=90, pady=5, ipady=7)
        self.drop_menu.grid(row=2, column=9, padx=0, pady=0, sticky=E)
        self.X_Y_label.grid(row=3, column=9, padx=5, pady=0, ipady=7,sticky=W)

        self.data_x_axis.grid(row=3, column=9, padx=150, pady=0, ipady=7, sticky=W)
        self.data_y_axis.grid(row=3, column=9, padx=0, pady=0, ipady=7,sticky=E)

        self.zero_vis.grid(row=4, column=9, padx=120, pady=5, ipady=5, sticky=W)
        self.zero_vis_label.grid(row=4, column=9, padx=85, pady=5, ipady=0, sticky=W)

        self.inf_vis.grid(row=4, column=9, padx=240, pady=5, ipady=5,sticky=W)
        self.inf_vis_label.grid(row=4, column=9, padx=205, pady=5, ipady=0,sticky=W)

        self.lamda.grid(row=4, column=9, padx=200, pady=5, ipady=5,sticky=E)
        self.lamda_label.grid(row=4, column=9, padx=250, pady=5, ipady=0,sticky=E)

        self.trans.grid(row=4, column=9, padx=100, pady=5, ipady=5,sticky=E)
        self.trans_label.grid(row=4, column=9, padx=150, pady=5, ipady=0,sticky=E)

        self.pw_indx.grid(row=4, column=9, padx=0, pady=5, ipady=5,sticky=E)
        self.pw_indx_label.grid(row=4, column=9, padx=50, pady=5, ipady=0,sticky=E)

        self.data_x_axis.bind("<Button-1>", lambda e: self.data_x_axis.delete(0.0, END))
        self.data_y_axis.bind("<Button-1>", lambda e: self.data_y_axis.delete(0.0, END))
        self.data_x_axis.insert(END, '  Enter shear rate') # END for Text and 0 for Entry
        self.data_y_axis.insert(END, '  Enter viscosity')

        self.zero_vis.bind("<Button-1>", lambda e: self.zero_vis.delete(0.0, END))
        self.inf_vis.bind("<Button-1>", lambda e: self.inf_vis.delete(0.0, END))
        self.lamda.bind("<Button-1>", lambda e: self.lamda.delete(0.0, END))
        self.trans.bind("<Button-1>", lambda e: self.trans.delete(0.0, END))
        self.pw_indx.bind("<Button-1>", lambda e: self.pw_indx.delete(0.0, END))

        self.submit_btn.grid(row=5, column=9, padx=200, pady=5,ipady=7,sticky=W)
        self.exit_btn.grid(row=5, column=9, padx=40, pady=5,ipady=7,sticky=E)
        self.data_x_axis.focus()
        self.data_y_axis.focus()

        self.zero_vis.focus()
        self.inf_vis.focus()
        self.lamda.focus()
        self.trans.focus()
        self.pw_indx.focus()

        self.make_textmenu()
        self.GUI.bind_class("Text", "<Button-3><ButtonRelease-3>", self.show_textmenu)
        self.GUI.bind_class("Text", "<Control-a>", self.callback_select_all)
        self.GUI.mainloop()

    def open_popup(self):
       self.pop= Toplevel(self.GUI)
       self.pop.geometry("300x100")
       self.pop.title("Report Window")
       Label(
                self.pop,text="Optimization failed." 
                " Check your data or\nChange fitting parameters.",
                font=('none 11 bold'),justify=LEFT).place(x=5,y=5
            )

    def close_window(self):
        self.GUI.destroy()
        exit()

    def make_textmenu(self):
        global m
        m = Menu(self.GUI, tearoff=0)
        m.add_command(label="Cut")
        m.add_command(label="Copy")
        m.add_command(label="Paste")
        m.add_separator()
        m.add_command(label="Select all")

    def callback_select_all(self, event):
        self.GUI.after(50, lambda:event.widget.select_range(0, 'end'))

    def show_textmenu(self, event):
        self.e_widget = event.widget
        m.entryconfigure("Cut",command=lambda: self.e_widget.event_generate("<<Cut>>"))
        m.entryconfigure("Copy",command=lambda: self.e_widget.event_generate("<<Copy>>"))
        m.entryconfigure("Paste",command=lambda: self.e_widget.event_generate("<<Paste>>"))
        m.entryconfigure("Select all",command=lambda: self.e_widget.select_range(0, 'end'))
        m.tk.call("tk_popup", m, event.x_root, event.y_root)

    def osPath(self):
        self.cwd = os.getcwd()
        os.path.filename = filedialog.askopenfilename(
                                                    parent=self.GUI,initialdir=self.cwd,
                                                    title='Please select a directory',
                                                    filetypes=(("text files", "*.txt"),
                                                    ("csv files", "*.csv"),("dat files", "*.dat*"))
                                                   )

        if os.path.filename:
            self.upload_txtentry.delete(0,END)
            self.upload_txtentry.insert(0,os.path.filename)
            os.path.filename=self.upload_txtentry.get()

            try:
                for i in np.loadtxt(os.path.filename):
                    self.shear_rate.append(i[0])
                    self.viscosity.append(i[1])

            except:
                try:
                    for i in np.loadtxt(os.path.filename,skiprows=2):
                        self.shear_rate.append(i[0])
                        self.viscosity.append(i[1])
                except:
                    self.open_popup()

    def PlotData(self):
        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μo = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μo = 0
        try:
            self.μf = float(self.inf_vis.get("1.0",'end-1c'))
        except:
            self.μf = 0
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = 0
        try:
            self.a  = float(self.trans.get("1.0",'end-1c'))
        except:
            self.a = 0
        try:
            self.n  = float(self.pw_indx.get("1.0",'end-1c'))
        except:
            self.n = 0
        if any(i != 0 for i in [self.μo,self.μf,self.λ,self.a,self.n]):
            try:
                self.FitModel(self.x,self.y,self.μo,self.μf,self.λ,self.a,self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitModel(self.x,self.y)
            except:
                self.open_popup()


if __name__ == "__main__":

    GraphicalUserInterface()
