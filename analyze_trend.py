import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import warnings
warnings.filterwarnings('ignore')

class IoTDataVisualizer:
    def __init__(self):
        self.intermediate_df = None
        self.blockchain_df = None
        self.merged_df = None
        self.current_metric = 'temperature'
        self.show_all_data = True
        self.show_blockchain_only = True
        self.show_legends = True
        
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("IoT Data Visualization & Blockchain Verification Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor('#f8f9fa')
        
        self.setup_gui()
        self.load_data()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File loading buttons
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Intermediate CSV", 
                  command=self.load_intermediate_file).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(file_frame, text="Load Blockchain CSV", 
                  command=self.load_blockchain_file).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Refresh Plot", 
                  command=self.update_plot).grid(row=0, column=2, padx=5)
        
        self.status_label = ttk.Label(file_frame, text="Status: Ready to load files", 
                                     foreground="blue")
        self.status_label.grid(row=0, column=3, padx=(20, 0))
        
        # Metric selection
        ttk.Label(control_frame, text="Select Metric:").grid(row=1, column=0, sticky=tk.W)
        self.metric_var = tk.StringVar(value='temperature')
        metric_combo = ttk.Combobox(control_frame, textvariable=self.metric_var,
                                   values=['temperature', 'humidity', 'light', 'voltage'],
                                   state='readonly', width=15)
        metric_combo.grid(row=1, column=1, padx=(10, 20), sticky=tk.W)
        metric_combo.bind('<<ComboboxSelected>>', self.on_metric_change)
        
        # Display options
        options_frame = ttk.Frame(control_frame)
        options_frame.grid(row=1, column=2, columnspan=2, padx=(20, 0))
        
        self.show_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show All Data Points", 
                       variable=self.show_all_var, 
                       command=self.update_plot).grid(row=0, column=0, padx=(0, 10))
        
        self.show_blockchain_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Highlight Blockchain Data", 
                       variable=self.show_blockchain_var, 
                       command=self.update_plot).grid(row=0, column=1, padx=(0, 10))
        
        self.show_legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Legend", 
                       variable=self.show_legend_var, 
                       command=self.update_plot).grid(row=0, column=2)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Data Statistics", padding="10")
        stats_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N), padx=(10, 0))
        
        self.stats_text = tk.Text(stats_frame, height=8, width=35, font=('Courier', 9))
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        # Plot frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add mouse wheel scroll support for zooming
        def on_scroll(event):
            if event.inaxes != self.ax:
                return
            
            # Get current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Calculate zoom factor
            zoom_factor = 1.1 if event.button == 'down' else 0.9
            
            # Get mouse position in data coordinates
            xdata, ydata = event.xdata, event.ydata
            
            # Calculate new limits
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            new_x_range = x_range * zoom_factor
            new_y_range = y_range * zoom_factor
            
            # Center zoom on mouse position
            new_xlim = [xdata - new_x_range * (xdata - xlim[0]) / x_range,
                       xdata + new_x_range * (xlim[1] - xdata) / x_range]
            new_ylim = [ydata - new_y_range * (ydata - ylim[0]) / y_range,
                       ydata + new_y_range * (ylim[1] - ydata) / y_range]
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.update_xaxis_ticks()
        
        # Connect scroll event
        self.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Add keyboard shortcuts for navigation
        def on_key_press(event):
            if event.key == 'r':  # Reset view
                self.reset_view()
            elif event.key == 'ctrl+z':  # Zoom out
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                x_range = xlim[1] - xlim[0]
                y_range = ylim[1] - ylim[0]
                self.ax.set_xlim([xlim[0] - x_range*0.1, xlim[1] + x_range*0.1])
                self.ax.set_ylim([ylim[0] - y_range*0.1, ylim[1] + y_range*0.1])
                self.update_xaxis_ticks()
        
        self.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Navigation toolbar with enhanced features
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Add custom zoom controls
        zoom_frame = ttk.Frame(toolbar_frame)
        zoom_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(zoom_frame, text="Reset View", 
                  command=self.reset_view, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", 
                  command=self.zoom_out, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Fit Data", 
                  command=self.fit_data, width=10).pack(side=tk.LEFT, padx=2)
        
    def update_xaxis_ticks(self):
        """Dynamically adjust x-axis ticks based on visible range"""
        if not hasattr(self, 'ax') or not hasattr(self, 'merged_df'):
            return
        
        # Get current x-axis limits
        xlim = self.ax.get_xlim()
        visible_min = mdates.num2date(xlim[0])
        visible_max = mdates.num2date(xlim[1])
        visible_range = visible_max - visible_min
        
        # Calculate appropriate tick intervals based on visible time range
        if visible_range.days > 30:
            # For months range
            self.ax.xaxis.set_major_locator(mdates.MonthLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif visible_range.days > 7:
            # For weeks range
            self.ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif visible_range.days > 1:
            # For days range
            self.ax.xaxis.set_major_locator(mdates.DayLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif visible_range.total_seconds() > 3600*12:
            # For hours range
            self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(visible_range.total_seconds()/3600/6))))
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            # For minutes/seconds range
            self.ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int(visible_range.total_seconds()/60/5))))
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Rotate labels for better readability
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Redraw the canvas
        self.canvas.draw()
    
    def load_data(self):
        """Load data from CSV files"""
        try:
            # Try to load default files
            self.load_intermediate_data('intermediate.csv')
            self.load_blockchain_data('blockchain.csv')
            self.merge_and_analyze_data()
            self.update_plot()
            
        except Exception as e:
            self.status_label.config(text="Status: Please load CSV files manually", 
                                   foreground="orange")
            print(f"Could not auto-load files: {e}")
    
    def load_intermediate_file(self):
        """Load intermediate CSV file via file dialog"""
        filename = filedialog.askopenfilename(
            title="Select Intermediate CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.load_intermediate_data(filename)
            self.merge_and_analyze_data()
            self.update_plot()
    
    def load_blockchain_file(self):
        """Load blockchain CSV file via file dialog"""
        filename = filedialog.askopenfilename(
            title="Select Blockchain CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.load_blockchain_data(filename)
            self.merge_and_analyze_data()
            self.update_plot()
    
    def load_intermediate_data(self, filename):
        """Load intermediate.csv data"""
        try:
            self.intermediate_df = pd.read_csv(filename)
            
            # Create datetime column
            self.intermediate_df['datetime'] = pd.to_datetime(
                self.intermediate_df['date'] + ' ' + self.intermediate_df['time'], 
                errors='coerce'
            )
            
            # Remove rows with invalid datetime
            self.intermediate_df = self.intermediate_df.dropna(subset=['datetime'])
            self.intermediate_df = self.intermediate_df.sort_values('datetime').reset_index(drop=True)
            
            print(f"✅ Loaded intermediate data: {len(self.intermediate_df)} records")
            self.status_label.config(text=f"Intermediate: {len(self.intermediate_df)} records loaded", 
                                   foreground="green")
            
        except Exception as e:
            error_msg = f"Error loading intermediate data: {e}"
            print(f"❌ {error_msg}")
            self.status_label.config(text=f"Error: {error_msg}", foreground="red")
            messagebox.showerror("Error", error_msg)
    
    def load_blockchain_data(self, filename):
        """Load blockchain.csv data"""
        try:
            self.blockchain_df = pd.read_csv(filename)
            
            # Create datetime column
            self.blockchain_df['datetime'] = pd.to_datetime(
                self.blockchain_df['date'] + ' ' + self.blockchain_df['time'], 
                errors='coerce'
            )
            
            # Remove rows with invalid datetime
            self.blockchain_df = self.blockchain_df.dropna(subset=['datetime'])
            self.blockchain_df = self.blockchain_df.sort_values('datetime').reset_index(drop=True)
            
            print(f"✅ Loaded blockchain data: {len(self.blockchain_df)} records")
            current_status = self.status_label.cget("text")
            self.status_label.config(text=f"{current_status} | Blockchain: {len(self.blockchain_df)} records", 
                                   foreground="green")
            
        except Exception as e:
            error_msg = f"Error loading blockchain data: {e}"
            print(f"❌ {error_msg}")
            self.status_label.config(text=f"Error: {error_msg}", foreground="red")
            messagebox.showerror("Error", error_msg)
    
    def merge_and_analyze_data(self):
        """Merge and analyze the datasets"""
        if self.intermediate_df is None or self.blockchain_df is None:
            return
        
        try:
            # Create a copy of intermediate data
            self.merged_df = self.intermediate_df.copy()
            
            # Add blockchain indicator
            self.merged_df['in_blockchain'] = False
            
            # Mark records that exist in blockchain
            for _, blockchain_row in self.blockchain_df.iterrows():
                # Find matching records in intermediate data
                matches = self.merged_df[
                    (self.merged_df['datetime'] == blockchain_row['datetime']) &
                    (self.merged_df['moteid'] == blockchain_row['moteid'])
                ]
                
                if not matches.empty:
                    self.merged_df.loc[matches.index, 'in_blockchain'] = True
            
            # Update statistics
            self.update_statistics()
            
            print(f"✅ Data merged successfully")
            
        except Exception as e:
            error_msg = f"Error merging data: {e}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def update_statistics(self):
        """Update the statistics display"""
        if self.merged_df is None:
            return
        
        try:
            total_intermediate = len(self.intermediate_df)
            total_blockchain = len(self.blockchain_df)
            blockchain_matches = self.merged_df['in_blockchain'].sum()
            
            # Analyze blockchain data by status/source
            blockchain_stats = ""
            if 'status' in self.blockchain_df.columns:
                status_counts = self.blockchain_df['status'].value_counts()
                blockchain_stats += "\nBlockchain by Status:\n"
                for status, count in status_counts.items():
                    blockchain_stats += f"  {status}: {count}\n"
            
            if 'source' in self.blockchain_df.columns:
                source_counts = self.blockchain_df['source'].value_counts()
                blockchain_stats += "\nBlockchain by Source:\n"
                for source, count in source_counts.items():
                    blockchain_stats += f"  {source}: {count}\n"
            
            # Calculate data reduction
            reduction_pct = ((total_intermediate - total_blockchain) / total_intermediate * 100) if total_intermediate > 0 else 0
            
            # Get current metric statistics
            metric = self.metric_var.get()
            if metric in self.merged_df.columns:
                intermediate_stats = self.intermediate_df[metric].describe()
                blockchain_stats_data = self.blockchain_df[metric].describe() if metric in self.blockchain_df.columns else None
                
                stats_text = f"""DATA OVERVIEW:
================
Intermediate Records: {total_intermediate:,}
Blockchain Records: {total_blockchain:,}
Matching Records: {blockchain_matches:,}
Data Reduction: {reduction_pct:.1f}%

{blockchain_stats}
CURRENT METRIC ({metric.upper()}):
================================
Intermediate Data:
  Count: {intermediate_stats['count']:.0f}
  Mean: {intermediate_stats['mean']:.2f}
  Std: {intermediate_stats['std']:.2f}
  Min: {intermediate_stats['min']:.2f}
  Max: {intermediate_stats['max']:.2f}

"""
                if blockchain_stats_data is not None:
                    stats_text += f"""Blockchain Data:
  Count: {blockchain_stats_data['count']:.0f}
  Mean: {blockchain_stats_data['mean']:.2f}
  Std: {blockchain_stats_data['std']:.2f}
  Min: {blockchain_stats_data['min']:.2f}
  Max: {blockchain_stats_data['max']:.2f}
"""
            else:
                stats_text = f"Select a valid metric to view statistics"
            
            # Update the text widget
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def on_metric_change(self, event=None):
        """Handle metric selection change"""
        self.current_metric = self.metric_var.get()
        self.update_statistics()
        self.update_plot()
    
    def update_plot(self):
        """Update the main plot"""
        if self.merged_df is None:
            return
        
        try:
            # Clear the plot
            self.ax.clear()
            
            metric = self.metric_var.get()
            
            if metric not in self.merged_df.columns:
                self.ax.text(0.5, 0.5, f"Metric '{metric}' not found in data", 
                           transform=self.ax.transAxes, ha='center', va='center',
                           fontsize=14, color='red')
                self.canvas.draw()
                return
            
            # Get display options
            show_all = self.show_all_var.get()
            show_blockchain = self.show_blockchain_var.get()
            show_legend = self.show_legend_var.get()
            
            # Plot all data points
            if show_all:
                # Non-blockchain data (green)
                non_blockchain = self.merged_df[~self.merged_df['in_blockchain']]
                if not non_blockchain.empty:
                    self.ax.scatter(non_blockchain['datetime'], non_blockchain[metric],
                                  c='green', alpha=0.6, s=20, label='Not in Blockchain',
                                  edgecolors='darkgreen', linewidths=0.5)
            
            # Plot blockchain data with different colors based on source/status
            if show_blockchain:
                blockchain_data = self.merged_df[self.merged_df['in_blockchain']]
                if not blockchain_data.empty:
                    # Get corresponding blockchain records to check source/status
                    anomaly_points = []
                    trend_points = []
                    other_points = []
                    
                    for _, row in blockchain_data.iterrows():
                        # Find matching record in blockchain_df to get source info
                        blockchain_match = self.blockchain_df[
                            (self.blockchain_df['datetime'] == row['datetime']) &
                            (self.blockchain_df['moteid'] == row['moteid'])
                        ]
                        
                        if not blockchain_match.empty:
                            source = blockchain_match.iloc[0].get('source', '')
                            status = blockchain_match.iloc[0].get('status', '')
                            
                            # Categorize based on source or status
                            if 'anomaly' in source.lower() or status == 'anomaly':
                                anomaly_points.append(row)
                            elif 'trend' in source.lower() or 'normal' in source.lower():
                                trend_points.append(row)
                            else:
                                other_points.append(row)
                        else:
                            other_points.append(row)
                    
                    # Plot anomaly points (red)
                    if anomaly_points:
                        anomaly_df = pd.DataFrame(anomaly_points)
                        self.ax.scatter(anomaly_df['datetime'], anomaly_df[metric],
                                      c='red', alpha=0.9, s=40, label='Anomaly Data (RF_first_anomaly)',
                                      edgecolors='darkred', linewidths=1.0, marker='o')
                    
                    # Plot trend points (blue)
                    if trend_points:
                        trend_df = pd.DataFrame(trend_points)
                        self.ax.scatter(trend_df['datetime'], trend_df[metric],
                                      c='blue', alpha=0.8, s=35, label='Trend Data (trend_normal)',
                                      edgecolors='darkblue', linewidths=0.8, marker='s')
                    
                    # Plot other blockchain points (orange)
                    if other_points:
                        other_df = pd.DataFrame(other_points)
                        self.ax.scatter(other_df['datetime'], other_df[metric],
                                      c='orange', alpha=0.8, s=30, label='Other Blockchain Data',
                                      edgecolors='darkorange', linewidths=0.8, marker='^')
            
            # Connect trend points with a blue line for trend visualization
            if show_blockchain:
                blockchain_data = self.merged_df[self.merged_df['in_blockchain']].sort_values('datetime')
                if len(blockchain_data) > 1:
                    # Separate trend points for line connection
                    trend_line_points = []
                    for _, row in blockchain_data.iterrows():
                        blockchain_match = self.blockchain_df[
                            (self.blockchain_df['datetime'] == row['datetime']) &
                            (self.blockchain_df['moteid'] == row['moteid'])
                        ]
                        if not blockchain_match.empty:
                            source = blockchain_match.iloc[0].get('source', '')
                            if 'trend' in source.lower() or 'normal' in source.lower():
                                trend_line_points.append(row)
                    
                    if len(trend_line_points) > 1:
                        trend_df = pd.DataFrame(trend_line_points)
                        self.ax.plot(trend_df['datetime'], trend_df[metric],
                                   color='blue', alpha=0.4, linewidth=2, linestyle='-',
                                   label='Trend Connection')
            
            # Formatting
            self.ax.set_xlabel('Time', fontsize=12)
            self.ax.set_ylabel(f'{metric.title()}', fontsize=12)
            self.ax.set_title(f'IoT Data Visualization: {metric.title()} vs Time\n'
                            f'Green = Not Stored | Red = Anomalies | Blue = Trends',
                            fontsize=14, pad=20)
            
            # Format x-axis dynamically
            self.update_xaxis_ticks()
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
            # Add legend
            if show_legend and (show_all or show_blockchain):
                self.ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
            
            # Add statistics text box with color-coded info
            total_points = len(self.merged_df)
            blockchain_points = self.merged_df['in_blockchain'].sum()
            
            # Count different types in blockchain
            anomaly_count = 0
            trend_count = 0
            if hasattr(self, 'blockchain_df') and self.blockchain_df is not None:
                if 'source' in self.blockchain_df.columns:
                    anomaly_count = self.blockchain_df['source'].str.contains('anomaly', case=False, na=False).sum()
                    trend_count = self.blockchain_df['source'].str.contains('trend', case=False, na=False).sum()
            
            stats_text = f'Total Points: {total_points}\nBlockchain Points: {blockchain_points}\nAnomalies: {anomaly_count} (Red)\nTrends: {trend_count} (Blue)\nReduction: {((total_points-blockchain_points)/total_points*100):.1f}%'
            
            self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontsize=10)
            
            # Enable interactive features
            self.ax.set_xlim(self.merged_df['datetime'].min(), self.merged_df['datetime'].max())
            
            # Tight layout and refresh
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            error_msg = f"Error updating plot: {e}"
            print(f"❌ {error_msg}")
            self.ax.text(0.5, 0.5, error_msg, transform=self.ax.transAxes, 
                        ha='center', va='center', fontsize=12, color='red')
            self.canvas.draw()
    
    def reset_view(self):
        """Reset the plot view to show all data"""
        if self.merged_df is not None:
            self.ax.relim()
            self.ax.autoscale()
            self.update_xaxis_ticks()
    
    def zoom_out(self):
        """Zoom out by 20%"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        self.ax.set_xlim([xlim[0] - x_range*0.1, xlim[1] + x_range*0.1])
        self.ax.set_ylim([ylim[0] - y_range*0.1, ylim[1] + y_range*0.1])
        self.update_xaxis_ticks()
    
    def fit_data(self):
        """Fit view to show all data points"""
        if self.merged_df is not None:
            metric = self.metric_var.get()
            if metric in self.merged_df.columns:
                self.ax.set_xlim(self.merged_df['datetime'].min(), self.merged_df['datetime'].max())
                self.ax.set_ylim(self.merged_df[metric].min(), self.merged_df[metric].max())
                self.update_xaxis_ticks()
    
    def run(self):
        """Start the application"""
        print("🚀 Starting IoT Data Visualization Tool...")
        print("📊 This tool helps verify blockchain data storage accuracy")
        print("🔍 Use zoom and pan tools to inspect data in detail")
        self.root.mainloop()

def main():
    """Main function to run the visualization tool"""
    print("="*70)
    print("🔬 IoT DATA VISUALIZATION & BLOCKCHAIN VERIFICATION TOOL")
    print("="*70)
    print()
    print("Features:")
    print("✅ Load and compare intermediate.csv vs blockchain.csv")
    print("✅ Interactive plotting with zoom and pan capabilities")
    print("✅ Multiple metrics: temperature, humidity, light, voltage")
    print("✅ Color-coded visualization: Green (not stored), Red (stored in blockchain)")
    print("✅ Real-time statistics and data analysis")
    print("✅ Trend line visualization for blockchain data")
    print("✅ Comprehensive data verification")
    print("✅ Dynamic x-axis tick adjustment for large datasets")
    print()
    print("Usage:")
    print("1. Load your intermediate.csv and blockchain.csv files")
    print("2. Select the metric you want to visualize")
    print("3. Use zoom/pan tools to inspect data points")
    print("4. Verify that correct trend points are stored in blockchain")
    print()
    print("Color Coding:")
    print("🟢 Green dots: Data NOT stored in blockchain")
    print("🔴 Red dots: ANOMALY data stored in blockchain (RF_first_anomaly)")
    print("🔵 Blue squares: TREND data stored in blockchain (trend_normal)")
    print("📈 Blue solid line: Trend data connection")
    print()
    print("Navigation:")
    print("🖱️ Mouse wheel: Zoom in/out at cursor position")
    print("⌨️ 'R' key: Reset view to show all data")
    print("⌨️ Ctrl+Z: Zoom out")
    print("🔧 Toolbar: Pan, zoom, save, and other tools")
    print("🎯 Custom buttons: Reset View, Zoom Out, Fit Data")
    print()
    
    # Create and run the application
    app = IoTDataVisualizer()
    app.run()

if __name__ == "__main__":
    main()