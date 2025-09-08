import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from db_config import get_connection
from openpyxl import Workbook
import matplotlib
matplotlib.use('TkAgg')
from PIL import ImageGrab
from tkcalendar import DateEntry

try:
    from ttkthemes import ThemedTk
except ImportError:
    ThemedTk = tk.Tk
    print("ttkthemes not found. Please install it to use themes.")

class CaneSummaryDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Cane Summary Dashboard")
        self.root.geometry("1600x900")
        
        # Set a modern theme if available
        if ThemedTk is not tk.Tk:
            try:
                self.root.set_theme("arc")  # Modern theme: "arc", "equilux", or "darkly"
            except Exception as e:
                print(f"Could not load selected theme: {e}. Falling back to default.")

        # Database Connection
        try:
            self.cnPay = get_connection()
        except (pyodbc.Error, FileNotFoundError) as e:
            messagebox.showerror("Database Connection Error", f"Could not connect to the database: {e}")
            self.root.destroy()
            return
        self.canetoday = 0.0

        # Main Layout using Grid for better responsiveness
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Header Section (Row 0)
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="ew")
        
        self.label_title = ttk.Label(header_frame, text="Daily Cane Summary", font=("Helvetica", 24, "bold"))
        self.label_title.pack(side="left", padx=(0, 50))

        date_frame = ttk.Frame(header_frame)
        date_frame.pack(side="left", padx=(0, 20))
        ttk.Label(date_frame, text="Select Date:", font=("Helvetica", 14)).pack(side="left", padx=5)
        self.date_picker = DateEntry(date_frame, selectmode='day', date_pattern='yyyy-mm-dd', font=("Helvetica", 12))
        self.date_picker.pack(side="left")
        self.date_picker.bind("<<DateEntrySelected>>", lambda event: self.bind_data())
        
        ttk.Button(header_frame, text="Refresh", command=self.bind_data).pack(side="left", padx=10)
        
        # Data & Charts Section (Row 1)
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Left Panel - Summary Cards & Data Grid
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Summary Cards (Grid layout for responsiveness)
        summary_cards_frame = ttk.Frame(left_panel)
        summary_cards_frame.pack(fill="x", pady=(0, 10))
        summary_cards_frame.columnconfigure((0, 1), weight=1)
        
        self.labels = {}
        self.create_summary_card(summary_cards_frame, "Total A (Count)", "a1sum", 0, 0)
        self.create_summary_card(summary_cards_frame, "Total B (Count)", "b1sum", 0, 1)
        self.create_summary_card(summary_cards_frame, "Total A (Tons)", "a2sum", 1, 0)
        self.create_summary_card(summary_cards_frame, "Total B (Tons)", "b2sum", 1, 1)

        # Treeview (Data Grid)
        data_grid_frame = ttk.LabelFrame(left_panel, text="Hourly Truck Data", padding="10")
        data_grid_frame.pack(fill="both", expand=True)

        columns = ("Time", "A_Count", "A_Tons", "B_Count", "B_Tons", "Total_Count", "Total_Tons")
        self.data_grid = ttk.Treeview(data_grid_frame, columns=columns, show="headings")
        self.data_grid.pack(fill="both", expand=True)

        for col in columns:
            self.data_grid.heading(col, text=col.replace('_', ' '), command=lambda _col=col: self.sort_treeview(_col, False))
            self.data_grid.column(col, anchor="center", width=120)

        # Right Panel - Charts Section
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self.panel_count = ttk.LabelFrame(right_panel, text="Truck Count by Line", padding="10")
        self.panel_count.pack(fill="both", expand=True, pady=(0, 10))
        
        self.panel_tons = ttk.LabelFrame(right_panel, text="Truck Tons by Line", padding="10")
        self.panel_tons.pack(fill="both", expand=True, pady=(10, 0))

        # Footer Section (Row 2)
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0), sticky="ew")

        # Export Buttons
        export_frame = ttk.Frame(footer_frame)
        export_frame.pack(side="left", padx=10)
        ttk.Button(export_frame, text="Export to Excel", command=self.export_to_excel).pack(side="left", padx=5)
        ttk.Button(export_frame, text="Export to Image", command=self.export_to_image).pack(side="left", padx=5)

        # Status Bar
        self.status_bar = ttk.Label(footer_frame, text="Ready.", relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(side="right", fill="x", expand=True)

        # Initial Data Load
        self.bind_data()
        self.root.after(30000, self.timer_tick)

    def create_summary_card(self, parent_frame, title, label_key, row, col):
        card = ttk.LabelFrame(parent_frame, text=title, padding="10")
        card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        parent_frame.grid_columnconfigure(col, weight=1)
        
        self.labels[label_key] = ttk.Label(card, text="0.00", font=("Helvetica", 20, "bold"))
        self.labels[label_key].pack(expand=True)

    def bind_data(self):
        self.status_bar.config(text="Loading data...")
        self.root.update_idletasks()
        try:
            selected_date = self.date_picker.get_date()
            mydate = datetime.combine(selected_date, datetime.min.time())
        except ValueError:
            messagebox.showerror("Invalid Date", "Please select a valid date.")
            self.status_bar.config(text="Error selecting date")
            return

        self.label_title.config(text=f"Daily Cane Summary for {selected_date.strftime('%A, %B %d, %Y')}")

        dt = pd.DataFrame(columns=["Time", "A_Count", "A_Tons", "B_Count", "B_Tons", "Total_Count", "Total_Tons"])
        end_time_today = mydate.replace(hour=18, minute=0, second=0)
        start_time_yesterday = end_time_today - timedelta(days=1)
        total_hours = 24
        
        a1_sum, a2_sum, b1_sum, b2_sum = 0, 0.0, 0, 0.0

        for i in range(total_hours):
            start_hour_interval = start_time_yesterday + timedelta(hours=i)
            end_hour_interval = start_hour_interval + timedelta(hours=1)
            only_time = f"{start_hour_interval.strftime('%H:%M')}"
            
            result_a1 = self.query_a_count(start_hour_interval, end_hour_interval)
            result_a2 = self.query_a_tons(start_hour_interval, end_hour_interval)
            result_b1 = self.query_b_count(start_hour_interval, end_hour_interval)
            result_b2 = self.query_b_tons(start_hour_interval, end_hour_interval)
            
            row_data = [only_time, result_a1, result_a2, result_b1, result_b2, result_a1 + result_b1, result_a2 + result_b2]
            dt.loc[len(dt)] = row_data

            a1_sum += result_a1
            a2_sum += result_a2
            b1_sum += result_b1
            b2_sum += result_b2

        self.labels["a1sum"].config(text=f"{a1_sum:,.0f}")
        self.labels["b1sum"].config(text=f"{b1_sum:,.0f}")
        self.labels["a2sum"].config(text=f"{a2_sum:,.2f}")
        self.labels["b2sum"].config(text=f"{b2_sum:,.2f}")

        # Populate Treeview
        for item in self.data_grid.get_children():
            self.data_grid.delete(item)
        for row in dt.itertuples(index=False):
            self.data_grid.insert("", "end", values=[
                row.Time, 
                f"{row.A_Count:,.0f}", 
                f"{row.A_Tons:,.2f}", 
                f"{row.B_Count:,.0f}", 
                f"{row.B_Tons:,.2f}", 
                f"{row.Total_Count:,.0f}", 
                f"{row.Total_Tons:,.2f}"
            ])
        
        # Create Charts
        self.create_chart_count(dt)
        self.create_chart_tons(dt)
        
        self.status_bar.config(text=f"Data loaded successfully at {datetime.now().strftime('%H:%M:%S')}")

    def sort_treeview(self, col, reverse):
        l = [(self.data_grid.set(k, col), k) for k in self.data_grid.get_children('')]
        l.sort(key=lambda t: float(t[0].replace(',', '')) if t[0].replace('.', '', 1).isdigit() else t[0], reverse=reverse)
        for index, (val, k) in enumerate(l):
            self.data_grid.move(k, '', index)
        self.data_grid.heading(col, command=lambda: self.sort_treeview(col, not reverse))

    def query_a_count(self, start_time, end_time):
        query = "SELECT COUNT(CAST(truck_q AS DECIMAL(18,2))) FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'A' AND (cane_type = '1' OR cane_type = '2') AND PRINT_W = 'y' AND WGT_NET > 0 AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')"
        try:
            with self.cnPay.cursor() as cursor:
                cursor.execute(query, (start_time, end_time))
                result = cursor.fetchone()[0]
                return float(result) if result else 0.0
        except pyodbc.Error as e:
            messagebox.showerror("Database Error", f"Query failed for A Count: {e}")
            return 0.0

    def query_a_tons(self, start_time, end_time):
        query = "SELECT SUM(CAST(wgt_net AS DECIMAL(18,2))) FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'A' AND (cane_type = '1' OR cane_type = '2') AND PRINT_W = 'y' AND WGT_NET > 0 AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')"
        try:
            with self.cnPay.cursor() as cursor:
                cursor.execute(query, (start_time, end_time))
                result = cursor.fetchone()[0]
                return float(result) if result else 0.0
        except pyodbc.Error as e:
            messagebox.showerror("Database Error", f"Query failed for A Tons: {e}")
            return 0.0

    def query_b_count(self, start_time, end_time):
        query = "SELECT COUNT(CAST(truck_q AS DECIMAL(18,2))) FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'B' AND (cane_type = '1' OR cane_type = '2') AND PRINT_W = 'y' AND WGT_NET > 0 AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')"
        try:
            with self.cnPay.cursor() as cursor:
                cursor.execute(query, (start_time, end_time))
                result = cursor.fetchone()[0]
                return float(result) if result else 0.0
        except pyodbc.Error as e:
            messagebox.showerror("Database Error", f"Query failed for B Count: {e}")
            return 0.0

    def query_b_tons(self, start_time, end_time):
        query = "SELECT SUM(CAST(wgt_net AS DECIMAL(18,2))) FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'B' AND (cane_type = '1' OR cane_type = '2') AND PRINT_W = 'y' AND WGT_NET > 0 AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')"
        try:
            with self.cnPay.cursor() as cursor:
                cursor.execute(query, (start_time, end_time))
                result = cursor.fetchone()[0]
                return float(result) if result else 0.0
        except pyodbc.Error as e:
            messagebox.showerror("Database Error", f"Query failed for B Tons: {e}")
            return 0.0

    def create_chart_count(self, dt):
        for widget in self.panel_count.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(6, 4))
        
        times = [row.Time for row in dt.itertuples(index=False)]
        values_a = [row.A_Count for row in dt.itertuples(index=False)]
        values_b = [row.B_Count for row in dt.itertuples(index=False)]
        
        x = range(len(times))
        width = 0.35
        ax.bar([i - width/2 for i in x], values_a, width, label='Line A', color="#1F77B4")
        ax.bar([i + width/2 for i in x], values_b, width, label='Line B', color="#2CA02C")

        ax.set_xticks(x)
        ax.set_xticklabels(times, rotation=45, ha="right")
        ax.set_xlabel("Time")
        ax.set_ylabel("Count (Trucks)")
        ax.set_title("Truck Count by Line", fontweight='bold')
        ax.legend()
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.panel_count)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_chart_tons(self, dt):
        for widget in self.panel_tons.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(6, 4))
        
        times = [row.Time for row in dt.itertuples(index=False)]
        values_a = [row.A_Tons for row in dt.itertuples(index=False)]
        values_b = [row.B_Tons for row in dt.itertuples(index=False)]
        
        x = range(len(times))
        width = 0.35
        ax.bar([i - width/2 for i in x], values_a, width, label='Line A', color="#FF7F0E")
        ax.bar([i + width/2 for i in x], values_b, width, label='Line B', color="#D62728")

        ax.set_xticks(x)
        ax.set_xticklabels(times, rotation=45, ha="right")
        ax.set_xlabel("Time")
        ax.set_ylabel("Weight (Tons)")
        ax.set_title("Truck Tons by Line", fontweight='bold')
        ax.legend()
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.panel_tons)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def export_to_excel(self):
        self.status_bar.config(text="Exporting to Excel...")
        self.root.update_idletasks()
        if not self.data_grid.get_children():
            messagebox.showerror("Error", "Export failed: No data available")
            self.status_bar.config(text="Export failed: No data")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Workbook", "*.xlsx")], initialfile="CaneSummary.xlsx")
        if file_path:
            try:
                columns = [self.data_grid.heading(col, "text") for col in self.data_grid["columns"]]
                data = [self.data_grid.item(item)["values"] for item in self.data_grid.get_children()]
                df = pd.DataFrame(data, columns=columns)
                df.to_excel(file_path, sheet_name="Cane Summary Report", index=False)
                messagebox.showinfo("Success", "Data exported to Excel successfully")
                self.status_bar.config(text="Excel export successful")
            except Exception as ex:
                messagebox.showerror("Error", str(ex))
                self.status_bar.config(text=f"Export failed: {ex}")

    def export_to_image(self):
        self.status_bar.config(text="Exporting image...")
        self.root.update_idletasks()
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")], initialfile="Dashboard.png")
        if file_path:
            try:
                x = self.root.winfo_rootx()
                y = self.root.winfo_rooty()
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                image.save(file_path)
                messagebox.showinfo("Success", "Screenshot exported successfully")
                self.status_bar.config(text="Image export successful")
            except Exception as ex:
                messagebox.showerror("Error", str(ex))
                self.status_bar.config(text=f"Export failed: {ex}")
    
    def timer_tick(self):
        current_date = datetime.now().date()
        selected_date = self.date_picker.get_date()
        if selected_date == current_date:
            self.bind_data()
        self.root.after(30000, self.timer_tick)

if __name__ == "__main__":
    root = ThemedTk()
    app = CaneSummaryDashboard(root)
    root.mainloop()
        # Populate Treeview

        for item in self.data_grid.get_children():

            self.data_grid.delete(item)

        for row in dt.itertuples(index=False):

            self.data_grid.insert("", "end", values=[

                row.Time, 

                f"{row.A_Count:,.0f}", 

                f"{row.A_Tons:,.2f}", 

                f"{row.B_Count:,.0f}", 

                f"{row.B_Tons:,.2f}", 

                f"{row.Total_Count:,.0f}", 

                f"{row.Total_Tons:,.2f}"

            ])

        

        # Create Charts

        self.create_chart_count(dt)

        self.create_chart_tons(dt)

        

        self.status_bar.config(text=f"Data loaded successfully at {datetime.now().strftime('%H:%M:%S')}")



    def sort_treeview(self, col, reverse):

        l = [(self.data_grid.set(k, col), k) for k in self.data_grid.get_children('')]

        l.sort(key=lambda t: float(t[0].replace(',', '')) if t[0].replace('.', '', 1).isdigit() else t[0], reverse=reverse)

        for index, (val, k) in enumerate(l):

            self.data_grid.move(k, '', index)

        self.data_grid.heading(col, command=lambda: self.sort_treeview(col, not reverse))



    def query_a_count(self, start_time, end_time):

        query = "SELECT COUNT(CAST(truck_q AS DECIMAL(18,2))) FROM cpc_truc WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'A'"

        try:

            with self.cnPay.cursor() as cursor:

                cursor.execute(query, (start_time, end_time))

                result = cursor.fetchone()[0]

                return float(result) if result else 0.0

        except pyodbc.Error as e:

            messagebox.showerror("Database Error", f"Query failed for A Count: {e}")

            return 0.0



    def query_a_tons(self, start_time, end_time):

        query = "SELECT SUM(CAST(wgt_net AS DECIMAL(18,2))) FROM cpc_truc WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'A'"

        try:

            with self.cnPay.cursor() as cursor:

                cursor.execute(query, (start_time, end_time))

                result = cursor.fetchone()[0]

                return float(result) if result else 0.0

        except pyodbc.Error as e:

            messagebox.showerror("Database Error", f"Query failed for A Tons: {e}")

            return 0.0



    def query_b_count(self, start_time, end_time):

        query = "SELECT COUNT(CAST(truck_q AS DECIMAL(18,2))) FROM cpc_truc WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'B'"

        try:

            with self.cnPay.cursor() as cursor:

                cursor.execute(query, (start_time, end_time))

                result = cursor.fetchone()[0]

                return float(result) if result else 0.0

        except pyodbc.Error as e:

            messagebox.showerror("Database Error", f"Query failed for B Count: {e}")

            return 0.0



    def query_b_tons(self, start_time, end_time):

        query = "SELECT SUM(CAST(wgt_net AS DECIMAL(18,2))) FROM cpc_truc WHERE print_q = '5' AND WGT_OUT_DT BETWEEN ? AND ? AND Prod_line = 'B'"

        try:

            with self.cnPay.cursor() as cursor:

                cursor.execute(query, (start_time, end_time))

                result = cursor.fetchone()[0]

                return float(result) if result else 0.0

        except pyodbc.Error as e:

            messagebox.showerror("Database Error", f"Query failed for B Tons: {e}")

            return 0.0



    def create_chart_count(self, dt):

        for widget in self.panel_count.winfo_children():

            widget.destroy()

            

        fig, ax = plt.subplots(figsize=(6, 4))

        

        times = [row.Time for row in dt.itertuples(index=False)]

        values_a = [row.A_Count for row in dt.itertuples(index=False)]

        values_b = [row.B_Count for row in dt.itertuples(index=False)]

        

        x = range(len(times))

        width = 0.35

        ax.bar([i - width/2 for i in x], values_a, width, label='Line A', color="#1F77B4")

        ax.bar([i + width/2 for i in x], values_b, width, label='Line B', color="#2CA02C")



        ax.set_xticks(x)

        ax.set_xticklabels(times, rotation=45, ha="right")

        ax.set_xlabel("Time")

        ax.set_ylabel("Count (Trucks)")

        ax.set_title("Truck Count by Line", fontweight='bold')

        ax.legend()

        plt.tight_layout()



        canvas = FigureCanvasTkAgg(fig, master=self.panel_count)

        canvas.draw()

        canvas.get_tk_widget().pack(fill="both", expand=True)



    def create_chart_tons(self, dt):

        for widget in self.panel_tons.winfo_children():

            widget.destroy()

            

        fig, ax = plt.subplots(figsize=(6, 4))

        

        times = [row.Time for row in dt.itertuples(index=False)]

        values_a = [row.A_Tons for row in dt.itertuples(index=False)]

        values_b = [row.B_Tons for row in dt.itertuples(index=False)]

        

        x = range(len(times))

        width = 0.35

        ax.bar([i - width/2 for i in x], values_a, width, label='Line A', color="#FF7F0E")

        ax.bar([i + width/2 for i in x], values_b, width, label='Line B', color="#D62728")



        ax.set_xticks(x)

        ax.set_xticklabels(times, rotation=45, ha="right")

        ax.set_xlabel("Time")

        ax.set_ylabel("Weight (Tons)")

        ax.set_title("Truck Tons by Line", fontweight='bold')

        ax.legend()

        plt.tight_layout()



        canvas = FigureCanvasTkAgg(fig, master=self.panel_tons)

        canvas.draw()

        canvas.get_tk_widget().pack(fill="both", expand=True)

    

    def export_to_excel(self):

        self.status_bar.config(text="Exporting to Excel...")

        self.root.update_idletasks()

        if not self.data_grid.get_children():

            messagebox.showerror("Error", "Export failed: No data available")

            self.status_bar.config(text="Export failed: No data")

            return



        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Workbook", "*.xlsx")], initialfile="CaneSummary.xlsx")

        if file_path:

            try:

                columns = [self.data_grid.heading(col, "text") for col in self.data_grid["columns"]]

                data = [self.data_grid.item(item)["values"] for item in self.data_grid.get_children()]

                df = pd.DataFrame(data, columns=columns)

                df.to_excel(file_path, sheet_name="Cane Summary Report", index=False)

                messagebox.showinfo("Success", "Data exported to Excel successfully")

                self.status_bar.config(text="Excel export successful")

            except Exception as ex:

                messagebox.showerror("Error", str(ex))

                self.status_bar.config(text=f"Export failed: {ex}")



    def export_to_image(self):

        self.status_bar.config(text="Exporting image...")

        self.root.update_idletasks()

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")], initialfile="Dashboard.png")

        if file_path:

            try:

                x = self.root.winfo_rootx()

                y = self.root.winfo_rooty()

                width = self.root.winfo_width()

                height = self.root.winfo_height()

                image = ImageGrab.grab(bbox=(x, y, x + width, y + height))

                image.save(file_path)

                messagebox.showinfo("Success", "Screenshot exported successfully")

                self.status_bar.config(text="Image export successful")

            except Exception as ex:

                messagebox.showerror("Error", str(ex))

                self.status_bar.config(text=f"Export failed: {ex}")

    

    def timer_tick(self):

        current_date = datetime.now().date()

        selected_date = self.date_picker.get_date()

        if selected_date == current_date:

            self.bind_data()

        self.root.after(30000, self.timer_tick)



if __name__ == "__main__":

    root = ThemedTk()

    app = CaneSummaryDashboard(root)

    root.mainloop()
