
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from datetime import datetime

CSV_FILE = "expenses.csv"
BUDGET_FILE = "budget.txt"

class ExpenseTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Expense Tracker")
        self.root.geometry("1100x750")
        self.root.configure(bg="#2b2b2b")

        self.setup_files()
        self.create_ui()
        self.load_expenses()
        self.load_daily_summary()
        self.update_summary()

    def setup_files(self):
        if not os.path.exists(CSV_FILE):
            pd.DataFrame(columns=["Date","Amount","Category","Description"]).to_csv(CSV_FILE,index=False)
        if not os.path.exists(BUDGET_FILE):
            open(BUDGET_FILE,"w").write("10000")

    def create_ui(self):
        tk.Label(self.root,text="PERSONAL EXPENSE TRACKER",
                 font=("Arial",20,"bold"),bg="#2b2b2b",fg="white").pack(pady=10)

        f=tk.Frame(self.root,bg="#2b2b2b"); f.pack()

        tk.Label(f,text="Amount",bg="#2b2b2b",fg="white").grid(row=0,column=0,padx=5,pady=5)
        self.amount=tk.Entry(f); self.amount.grid(row=0,column=1)

        tk.Label(f,text="Category",bg="#2b2b2b",fg="white").grid(row=0,column=2)
        self.category=ttk.Combobox(f,values=["Food","Travel","Shopping","Bills","Entertainment","Other"])
        self.category.grid(row=0,column=3); self.category.current(0)

        tk.Label(f,text="Date",bg="#2b2b2b",fg="white").grid(row=0,column=4)
        self.date=DateEntry(f,date_pattern="yyyy-mm-dd")
        self.date.grid(row=0,column=5)

        tk.Label(f,text="Description",bg="#2b2b2b",fg="white").grid(row=1,column=0)
        self.desc=tk.Entry(f,width=35); self.desc.grid(row=1,column=1,columnspan=2)

        tk.Label(f,text="Budget",bg="#2b2b2b",fg="white").grid(row=1,column=3)
        self.budget=tk.Entry(f)
        self.budget.insert(0,open(BUDGET_FILE).read())
        self.budget.grid(row=1,column=4)

        bf=tk.Frame(self.root,bg="#2b2b2b"); bf.pack(pady=10)
        tk.Button(bf,text="Add Expense",command=self.add_expense).grid(row=0,column=0,padx=5)
        tk.Button(bf,text="Delete Selected",command=self.delete_expense).grid(row=0,column=1,padx=5)
        tk.Button(bf,text="Show Charts",command=self.show_charts).grid(row=0,column=2,padx=5)
        tk.Button(bf,text="Export Report",command=self.export_report).grid(row=0,column=3,padx=5)

        sf=tk.Frame(self.root,bg="#2b2b2b"); sf.pack()
        self.filter=ttk.Combobox(sf,values=["All","Food","Travel","Shopping","Bills","Entertainment","Other"])
        self.filter.current(0)
        self.filter.pack(side="left",padx=5)
        tk.Button(sf,text="Search",command=self.search_category).pack(side="left")

        self.total=tk.Label(self.root,bg="#2b2b2b",fg="white")
        self.total.pack()
        self.remaining=tk.Label(self.root,bg="#2b2b2b",fg="white")
        self.remaining.pack()
        self.today=tk.Label(self.root,bg="#2b2b2b",fg="white")
        self.today.pack()

        cols=("Date","Amount","Category","Description")
        self.tree=ttk.Treeview(self.root,columns=cols,show="headings",height=12)
        for c in cols:
            self.tree.heading(c,text=c, anchor = "center")
            self.tree.column(c, anchor= "center", width = 250)
        self.tree.pack(fill="x",padx=10,pady=10)

        tk.Label(self.root,text="Daily Summary",font=("Arial",12,"bold"),
                 bg="#2b2b2b",fg="white").pack()

        self.daily=ttk.Treeview(self.root,columns=("Date","Total"),show="headings",height=6)
        self.daily.heading("Date",text="Date")
        self.daily.heading("Total",text="Total Spent")
        self.daily.column("Date", anchor="center", width=400)
        self.daily.column("Total", anchor="center", width=400)
        self.daily.pack(fill="x",padx=10)

    def add_expense(self):
        try:
            amt=float(self.amount.get())
        except:
            messagebox.showerror("Error","Enter valid amount")
            return

        pd.DataFrame([{
            "Date":self.date.get(),
            "Amount":amt,
            "Category":self.category.get(),
            "Description":self.desc.get()
        }]).to_csv(CSV_FILE,mode="a",header=False,index=False)

        open(BUDGET_FILE,"w").write(self.budget.get())
        self.load_expenses(); self.load_daily_summary(); self.update_summary()
        self.amount.delete(0,"end"); self.desc.delete(0,"end")

    def load_expenses(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        df=pd.read_csv(CSV_FILE)
        for _,r in df.iterrows():
            self.tree.insert("", "end", values=list(r))

    def load_daily_summary(self):
        for i in self.daily.get_children(): self.daily.delete(i)
        df=pd.read_csv(CSV_FILE)
        if df.empty: return
        s=df.groupby("Date")["Amount"].sum().reset_index().sort_values("Date",ascending=False)
        for _,r in s.iterrows():
            self.daily.insert("", "end", values=(r["Date"], f"₹{r['Amount']:.2f}"))

    def update_summary(self):
        df=pd.read_csv(CSV_FILE)
        total=df["Amount"].sum() if not df.empty else 0
        budget=float(self.budget.get() or 0)
        rem=budget-total
        today=datetime.now().strftime("%Y-%m-%d")
        today_total=df[df["Date"]==today]["Amount"].sum() if not df.empty else 0

        self.total.config(text=f"Total Spent: ₹{total:.2f}")
        self.remaining.config(text=f"Remaining Budget: ₹{rem:.2f}")
        self.today.config(text=f"Today's Spending: ₹{today_total:.2f}")

        if budget > 0 and total >= budget*0.8 and total < budget:
            messagebox.showwarning("Warning","80% of budget used")

    def delete_expense(self):
        sel=self.tree.selection()
        if not sel: return
        vals=self.tree.item(sel[0])["values"]
        df=pd.read_csv(CSV_FILE)
        mask=(df["Date"]==vals[0])&(df["Amount"]==float(vals[1]))&(df["Category"]==vals[2])&(df["Description"]==vals[3])
        df=df[~mask]
        df.to_csv(CSV_FILE,index=False)
        self.load_expenses(); self.load_daily_summary(); self.update_summary()

    def search_category(self):
        cat=self.filter.get()
        df=pd.read_csv(CSV_FILE)
        if cat!="All":
            df=df[df["Category"]==cat]
        for i in self.tree.get_children(): self.tree.delete(i)
        for _,r in df.iterrows():
            self.tree.insert("", "end", values=list(r))

    def show_charts(self):
        df=pd.read_csv(CSV_FILE)
        if df.empty:
            return

        cat=df.groupby("Category")["Amount"].sum()
        plt.figure()
        plt.pie(cat,labels=cat.index,autopct="%1.1f%%")
        plt.title("Expense Distribution")

        plt.figure()
        cat.plot(kind="bar")
        plt.title("Category Wise Spending")

        plt.figure()
        df.groupby("Date")["Amount"].sum().plot(marker="o")
        plt.title("Daily Spending Trend")

        plt.show()

    def export_report(self):
        path=filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            pd.read_csv(CSV_FILE).to_csv(path,index=False)
            messagebox.showinfo("Success","Report exported")

root=tk.Tk()
ExpenseTracker(root)
root.mainloop()
