"""
NCJM Stockflow - Complete Offline Inventory Management System
Version 2.0 - Enterprise Edition
All Features Included: Barcode, Receipts, Analytics, Multi-user, etc.
"""

import sqlite3
import csv
import json
import os
import shutil
from datetime import datetime, date, timedelta
import io
import base64

# Kivy imports
from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, NumericProperty, ListProperty, BooleanProperty, ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.filechooser import FileChooserIconView
from kivy.graphics import Color, Rectangle, RoundedRectangle, Line

# PDF Generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.utils import ImageReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠ ReportLab not installed. PDF receipts disabled.")

# Barcode Scanner
try:
    from pyzbar import pyzbar
    import cv2
    BARCODE_AVAILABLE = True
except ImportError:
    BARCODE_AVAILABLE = False
    print("⚠ pyzbar/cv2 not installed. Barcode scanner disabled.")

# Charts & Analytics
try:
    from kivy.garden.matplotlib import FigureCanvasKivyAgg
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("⚠ matplotlib not installed. Charts disabled.")

# Forecasting
try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False
    print("⚠ sklearn not installed. Sales forecasting disabled.")


# ========== CONFIGURATION ==========
DB_NAME = "users.db"
EXPORT_DIR = "/storage/emulated/0/BookKeep/exports"
BACKUP_DIR = "/storage/emulated/0/BookKeep/backups"
RECEIPT_DIR = "/storage/emulated/0/BookKeep/receipts"
PHOTO_DIR = "/storage/emulated/0/BookKeep/photos"
REPORT_DIR = "/storage/emulated/0/BookKeep/reports"

# Create directories
for directory in [EXPORT_DIR, BACKUP_DIR, RECEIPT_DIR, PHOTO_DIR, REPORT_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Directory creation error: {e}")


# ========== UTILITY FUNCTIONS ==========

def log_action(action, detail, user="system"):
    """Enhanced logging with user tracking"""
    try:
        with open("app.log", "a") as f:
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{user} | {action} | {detail}\n"
            )
    except Exception as e:
        print(f"Logging error: {e}")


def safe_play_sound(filename):
    """Safely play sound without crashing"""
    try:
        from kivy.core.audio import SoundLoader
        if os.path.exists(filename):
            sound = SoundLoader.load(filename)
            if sound:
                sound.play()
    except Exception as e:
        print(f"Sound error: {e}")


def format_currency(amount):
    """Format number as currency"""
    try:
        return f"₱{float(amount):,.2f}"
    except:
        return "₱0.00"


def validate_date(date_str):
    """Validate YYYY-MM-DD date format"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except:
        return False


def auto_backup():
    """Automatic database backup"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"backup_{timestamp}.db")
        shutil.copy2(DB_NAME, backup_file)
        
        # Keep only last 7 backups
        backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.db')])
        if len(backups) > 7:
            for old_backup in backups[:-7]:
                os.remove(os.path.join(BACKUP_DIR, old_backup))
        
        log_action("AUTO_BACKUP", f"Database backed up to {backup_file}")
        return True
    except Exception as e:
        print(f"Backup error: {e}")
        return False


# ========== DATABASE FUNCTIONS ==========

def init_database():
    """Initialize all database tables"""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            
            # Users table
            c.execute("""
                CREATE TABLE IF NOT EXISTS users(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'admin',
                    created_date TEXT,
                    last_login TEXT
                )
            """)
            
            # Inventory table
            c.execute("""
                CREATE TABLE IF NOT EXISTS inventory(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    barcode TEXT,
                    purchase_date TEXT,
                    purchased_stock INTEGER DEFAULT 0,
                    cost_price REAL DEFAULT 0,
                    color TEXT DEFAULT '0.2,0.6,0.86,1',
                    photo_path TEXT,
                    expiry_date TEXT,
                    location TEXT DEFAULT 'main',
                    supplier_id INTEGER,
                    created_by TEXT,
                    UNIQUE(category, location)
                )
            """)
            
            # Transactions table
            c.execute("""
                CREATE TABLE IF NOT EXISTS transactions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT DEFAULT 'sale',
                    category TEXT,
                    amount REAL,
                    date TEXT,
                    time TEXT,
                    quantity INTEGER,
                    sell_price REAL,
                    profit REAL,
                    payment_method TEXT DEFAULT 'cash',
                    discount REAL DEFAULT 0,
                    customer_id INTEGER,
                    user_id TEXT,
                    location TEXT DEFAULT 'main'
                )
            """)
            
            # Customers table
            c.execute("""
                CREATE TABLE IF NOT EXISTS customers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    loyalty_points INTEGER DEFAULT 0,
                    total_purchases REAL DEFAULT 0,
                    created_date TEXT,
                    last_purchase_date TEXT
                )
            """)
            
            # Suppliers table
            c.execute("""
                CREATE TABLE IF NOT EXISTS suppliers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    contact_person TEXT,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    total_orders REAL DEFAULT 0,
                    created_date TEXT
                )
            """)
            
            # Product variants table
            c.execute("""
                CREATE TABLE IF NOT EXISTS product_variants(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    variant_name TEXT,
                    variant_type TEXT,
                    stock INTEGER DEFAULT 0,
                    price_modifier REAL DEFAULT 0,
                    UNIQUE(category, variant_name)
                )
            """)
            
            # Promotions table
            c.execute("""
                CREATE TABLE IF NOT EXISTS promotions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    type TEXT,
                    category TEXT,
                    discount_percent REAL,
                    discount_amount REAL,
                    start_date TEXT,
                    end_date TEXT,
                    active INTEGER DEFAULT 1
                )
            """)
            
            # Layaway table
            c.execute("""
                CREATE TABLE IF NOT EXISTS layaway(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER,
                    transaction_id INTEGER,
                    total_amount REAL,
                    paid_amount REAL DEFAULT 0,
                    balance REAL,
                    due_date TEXT,
                    status TEXT DEFAULT 'active',
                    created_date TEXT
                )
            """)
            
            # Audit log table
            c.execute("""
                CREATE TABLE IF NOT EXISTS audit_log(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT,
                    action TEXT,
                    details TEXT,
                    timestamp TEXT
                )
            """)
            
            # Settings table
            c.execute("""
                CREATE TABLE IF NOT EXISTS settings(
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Migrations
            try:
                c.execute("SELECT barcode FROM inventory LIMIT 1")
            except sqlite3.OperationalError:
                c.execute("ALTER TABLE inventory ADD COLUMN barcode TEXT")
                c.execute("ALTER TABLE inventory ADD COLUMN photo_path TEXT")
                c.execute("ALTER TABLE inventory ADD COLUMN expiry_date TEXT")
                c.execute("ALTER TABLE inventory ADD COLUMN location TEXT DEFAULT 'main'")
                c.execute("ALTER TABLE inventory ADD COLUMN supplier_id INTEGER")
                c.execute("ALTER TABLE inventory ADD COLUMN created_by TEXT")
                print("✓ Migrated inventory table")
            
            try:
                c.execute("SELECT payment_method FROM transactions LIMIT 1")
            except sqlite3.OperationalError:
                c.execute("ALTER TABLE transactions ADD COLUMN payment_method TEXT DEFAULT 'cash'")
                c.execute("ALTER TABLE transactions ADD COLUMN discount REAL DEFAULT 0")
                c.execute("ALTER TABLE transactions ADD COLUMN customer_id INTEGER")
                c.execute("ALTER TABLE transactions ADD COLUMN user_id TEXT")
                c.execute("ALTER TABLE transactions ADD COLUMN location TEXT DEFAULT 'main'")
                c.execute("ALTER TABLE transactions ADD COLUMN time TEXT")
                print("✓ Migrated transactions table")
            
            # Insert default settings
            c.execute("INSERT OR IGNORE INTO settings VALUES('theme', 'dark')")
            c.execute("INSERT OR IGNORE INTO settings VALUES('language', 'en')")
            c.execute("INSERT OR IGNORE INTO settings VALUES('currency', '₱')")
            c.execute("INSERT OR IGNORE INTO settings VALUES('tax_rate', '0')")
            c.execute("INSERT OR IGNORE INTO settings VALUES('store_name', 'NCJM Stockflow')")
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Database init error: {e}")
        return False


# ========== BARCODE SCANNER ==========

class BarcodeScannerPopup(Popup):
    """Barcode scanner using camera"""
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.title = "Scan Barcode"
        self.size_hint = (0.9, 0.8)
        
        layout = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        
        if BARCODE_AVAILABLE:
            self.camera = Camera(resolution=(640, 480), play=True)
            layout.add_widget(self.camera)
            
            btn_layout = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
            btn_scan = Button(text="📷 Scan", on_release=self.scan_barcode)
            btn_cancel = Button(text="Cancel", on_release=self.dismiss)
            btn_layout.add_widget(btn_scan)
            btn_layout.add_widget(btn_cancel)
            layout.add_widget(btn_layout)
            
            Clock.schedule_interval(self.auto_scan, 1.0)
        else:
            layout.add_widget(Label(
                text="Barcode scanner not available.\nInstall: pip install pyzbar opencv-python"
            ))
            layout.add_widget(Button(text="Close", on_release=self.dismiss))
        
        self.content = layout
    
    def auto_scan(self, dt):
        """Auto-scan every second"""
        if BARCODE_AVAILABLE and hasattr(self, 'camera'):
            self.scan_barcode(None)
    
    def scan_barcode(self, instance):
        """Scan barcode from camera"""
        try:
            if not BARCODE_AVAILABLE:
                return
            
            texture = self.camera.texture
            if texture:
                pixels = texture.pixels
                size = texture.size
                
                # Convert to numpy array
                img_array = np.frombuffer(pixels, dtype=np.uint8)
                img_array = img_array.reshape(size[1], size[0], 4)
                
                # Convert RGBA to RGB
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                # Decode barcodes
                barcodes = pyzbar.decode(img_rgb)
                
                if barcodes:
                    barcode_data = barcodes[0].data.decode('utf-8')
                    Clock.unschedule(self.auto_scan)
                    self.dismiss()
                    self.callback(barcode_data)
        except Exception as e:
            print(f"Barcode scan error: {e}")
    
    def on_dismiss(self):
        if hasattr(self, 'camera'):
            self.camera.play = False
        Clock.unschedule(self.auto_scan)


# ========== RECEIPT GENERATOR ==========

class ReceiptGenerator:
    """Generate PDF receipts"""
    
    @staticmethod
    def generate(transaction_data, filename=None):
        """Generate PDF receipt"""
        if not PDF_AVAILABLE:
            print("PDF generation not available")
            return None
        
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(RECEIPT_DIR, f"receipt_{timestamp}.pdf")
            
            c = pdf_canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            
            # Header
            c.setFont("Helvetica-Bold", 20)
            c.drawString(inch, height - inch, "NCJM STOCKFLOW")
            
            c.setFont("Helvetica", 10)
            c.drawString(inch, height - inch - 20, "Offline D-Inventory System")
            c.drawString(inch, height - inch - 35, f"Date: {transaction_data.get('date', '')}")
            c.drawString(inch, height - inch - 50, f"Time: {transaction_data.get('time', '')}")
            c.drawString(inch, height - inch - 65, f"Receipt #: {transaction_data.get('id', '')}")
            
            # Line
            c.line(inch, height - inch - 80, width - inch, height - inch - 80)
            
            # Items
            y = height - inch - 110
            c.setFont("Helvetica-Bold", 12)
            c.drawString(inch, y, "Item")
            c.drawString(3*inch, y, "Qty")
            c.drawString(4*inch, y, "Price")
            c.drawString(5*inch, y, "Total")
            
            y -= 20
            c.setFont("Helvetica", 11)
            c.drawString(inch, y, transaction_data.get('category', ''))
            c.drawString(3*inch, y, str(transaction_data.get('quantity', '')))
            c.drawString(4*inch, y, format_currency(transaction_data.get('sell_price', 0)))
            c.drawString(5*inch, y, format_currency(transaction_data.get('amount', 0)))
            
            # Line
            y -= 20
            c.line(inch, y, width - inch, y)
            
            # Totals
            y -= 25
            c.setFont("Helvetica-Bold", 12)
            subtotal = transaction_data.get('amount', 0)
            discount = transaction_data.get('discount', 0)
            total = subtotal - discount
            
            c.drawString(4*inch, y, "Subtotal:")
            c.drawString(5*inch, y, format_currency(subtotal))
            
            if discount > 0:
                y -= 20
                c.drawString(4*inch, y, "Discount:")
                c.drawString(5*inch, y, f"-{format_currency(discount)}")
            
            y -= 25
            c.setFont("Helvetica-Bold", 14)
            c.drawString(4*inch, y, "TOTAL:")
            c.drawString(5*inch, y, format_currency(total))
            
            # Footer
            c.setFont("Helvetica", 9)
            c.drawString(inch, inch, "Thank you for your business!")
            c.drawString(inch, inch - 15, f"Payment: {transaction_data.get('payment_method', 'Cash')}")
            
            c.save()
            log_action("RECEIPT_GENERATED", f"Receipt saved: {filename}")
            return filename
        except Exception as e:
            print(f"Receipt generation error: {e}")
            return None


# ========== CHART GENERATOR ==========

class ChartGenerator:
    """Generate charts and graphs"""
    
    @staticmethod
    def generate_profit_chart(data, save_path=None):
        """Generate profit line chart"""
        if not CHARTS_AVAILABLE:
            print("Charts not available")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            dates = [d[0] for d in data]
            profits = [d[1] for d in data]
            
            ax.plot(dates, profits, marker='o', linewidth=2, color='#2196F3')
            ax.fill_between(dates, profits, alpha=0.3, color='#2196F3')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Profit (₱)', fontsize=12)
            ax.set_title('Daily Profit Trend', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
        except Exception as e:
            print(f"Chart generation error: {e}")
            return None
    
    @staticmethod
    def generate_sales_pie_chart(data, save_path=None):
        """Generate sales distribution pie chart"""
        if not CHARTS_AVAILABLE:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            categories = [d[0] for d in data]
            sales = [d[1] for d in data]
            
            colors = plt.cm.Set3(range(len(categories)))
            ax.pie(sales, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Sales by Category', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
        except Exception as e:
            print(f"Chart error: {e}")
            return None


# ========== FORECASTING ENGINE ==========

class SalesForecaster:
    """AI-powered sales forecasting"""
    
    @staticmethod
    def forecast_next_week(category):
        """Predict next week's sales"""
        if not FORECASTING_AVAILABLE:
            return None
        
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT date, SUM(quantity) as qty
                    FROM transactions
                    WHERE category = ? AND date >= date('now', '-60 days')
                    GROUP BY date
                    ORDER BY date
                """, (category,))
                data = c.fetchall()
            
            if len(data) < 7:
                return None
            
            # Prepare data
            X = np.array(range(len(data))).reshape(-1, 1)
            y = np.array([d[1] for d in data])
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict next 7 days
            future_X = np.array(range(len(data), len(data) + 7)).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            return {
                'next_week_total': int(sum(predictions)),
                'daily_average': int(np.mean(predictions)),
                'trend': 'up' if model.coef_[0] > 0 else 'down'
            }
        except Exception as e:
            print(f"Forecast error: {e}")
            return None


# ========== AUTHENTICATION SCREENS ==========

class LoginScreen(Screen):
    def do_login(self, username, password):
        if not username or not password:
            Popup(
                title="Error",
                content=Label(text="Username and Password required"),
                size_hint=(0.6, 0.4),
            ).open()
            return

        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT id, role FROM users WHERE username=? AND password=?",
                    (username, password),
                )
                row = c.fetchone()

            if row:
                user_id, role = row
                
                # Update last login
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        "UPDATE users SET last_login=? WHERE username=?",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username)
                    )
                    conn.commit()
                
                log_action("LOGIN", f"{username} logged in", username)
                self.manager.admin_name = username
                self.manager.admin_role = role
                self.manager.admin_id = user_id

                self.manager.current = "loading"
                self.manager.get_screen("loading").start_loading(
                    next_screen="home", message="Logging in..."
                )
            else:
                Popup(
                    title="Login Failed",
                    content=Label(text="Invalid username or password"),
                    size_hint=(0.6, 0.4),
                ).open()
        except Exception as e:
            Popup(
                title="Database Error",
                content=Label(text=f"Login error: {str(e)}"),
                size_hint=(0.7, 0.4),
            ).open()
            print(f"Login error: {e}")


class RegisterScreen(Screen):
    def do_register(self, username, password):
        if not username or not password:
            Popup(
                title="Error",
                content=Label(text="Username and Password required"),
                size_hint=(0.6, 0.4),
            ).open()
            return

        if len(password) < 8:
            Popup(
                title="Error",
                content=Label(text=f"Password must be at least 8 characters\n(You entered {len(password)} chars)"),
                size_hint=(0.6, 0.4),
            ).open()
            return

        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                
                # Check if any user already exists
                c.execute("SELECT COUNT(*) FROM users")
                user_count = c.fetchone()[0]
                
                if user_count >= 1:
                    Popup(
                        title="Registration Closed",
                        content=Label(
                            text="⚠ Only 1 admin allowed!\n\nAn admin already exists.\nContact admin to delete account first.",
                            halign="center"
                        ),
                        size_hint=(0.7, 0.5),
                    ).open()
                    return
                
                # Insert new user
                c.execute(
                    "INSERT INTO users(username, password, role, created_date) VALUES(?, ?, 'admin', ?)",
                    (username, password, datetime.now().strftime("%Y-%m-%d")),
                )
                conn.commit()
                
                log_action("REGISTER", f"{username} registered")
                Popup(
                    title="✓ Success",
                    content=Label(text="Admin registered successfully!\n\nYou can now log in."),
                    size_hint=(0.6, 0.4),
                ).open()
                
                self.ids.username.text = ""
                self.ids.password.text = ""
                
                Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'login'), 1.5)
                
        except sqlite3.IntegrityError:
            Popup(
                title="Error",
                content=Label(text="Username already exists!"),
                size_hint=(0.7, 0.5),
            ).open()
        except Exception as e:
            Popup(
                title="Database Error",
                content=Label(text=f"Error: {str(e)}"),
                size_hint=(0.7, 0.5),
            ).open()
            print(f"Registration error: {e}")


# ========== HOME SCREEN (ENHANCED) ==========

class HomeScreen(Screen):
    selected_category = StringProperty("")
    search_text = StringProperty("")

    def on_enter(self):
        self.load_categories()
        self.update_welcome()
        self.load_low_stock()

    def update_welcome(self):
        if hasattr(self.manager, "admin_name") and self.manager.admin_name:
            try:
                self.ids.sidebar.ids.admin_label.text = (
                    f"Welcome, {self.manager.admin_name}!"
                )
            except (KeyError, AttributeError):
                pass

    def filter_categories(self, search_text):
        self.search_text = search_text.lower()
        self.load_categories()

    def load_low_stock(self):
        """Load low stock items"""
        try:
            low_stock_grid = self.ids.low_stock_grid
            low_stock_grid.clear_widgets()

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT i.category, i.purchased_stock, 
                           COALESCE(SUM(t.quantity), 0) as sold,
                           i.expiry_date
                    FROM inventory i
                    LEFT JOIN transactions t ON i.category = t.category
                    GROUP BY i.category
                    HAVING (i.purchased_stock - sold) < 10
                    ORDER BY (i.purchased_stock - sold) ASC
                    """
                )
                rows = c.fetchall()

            if not rows:
                low_stock_grid.add_widget(Label(
                    text="All items well stocked ✓",
                    color=(0.5, 0.9, 0.5, 1),
                    size_hint_y=None,
                    height=dp(30),
                    font_size=sp(13)
                ))
                return

            for category, purchased, sold, expiry in rows:
                remaining = (purchased or 0) - (sold or 0)
                
                # Check expiry
                warning = ""
                if expiry:
                    try:
                        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
                        days_to_expiry = (expiry_date - datetime.now()).days
                        if days_to_expiry < 7:
                            warning = f" (expires in {days_to_expiry} days!)"
                    except:
                        pass
                
                color = (1, 0.3, 0.3, 1) if remaining < 5 else (1, 0.8, 0.2, 1)
                
                btn = Button(
                    text=f"⚠ {category}: {remaining} left{warning}",
                    size_hint_y=None,
                    height=dp(40),
                    background_normal="",
                    background_color=color,
                    color=(1, 1, 1, 1),
                    font_size=sp(12)
                )
                btn.bind(on_release=lambda inst, name=category: self.open_purchase_popup(name))
                low_stock_grid.add_widget(btn)
        except Exception as e:
            print(f"Load low stock error: {e}")

    def load_categories(self):
        try:
            grid = self.ids.category_grid
            grid.clear_widgets()

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT i.category, i.purchased_stock, i.cost_price, i.color,
                           COALESCE(SUM(t.quantity), 0) as sold, i.photo_path, i.barcode
                    FROM inventory i
                    LEFT JOIN transactions t ON i.category = t.category
                    GROUP BY i.category
                    ORDER BY i.category
                    """
                )
                rows = c.fetchall()

            for category, purchased_stock, cost_price, color, sold, photo_path, barcode in rows:
                if self.search_text and self.search_text not in category.lower():
                    continue

                remaining = (purchased_stock or 0) - (sold or 0)
                
                row = BoxLayout(
                    orientation="horizontal",
                    size_hint_y=None,
                    height=dp(60),
                    spacing=dp(6),
                )

                # Product photo (if available)
                if photo_path and os.path.exists(photo_path):
                    img = Image(
                        source=photo_path,
                        size_hint_x=None,
                        width=dp(60)
                    )
                    row.add_widget(img)

                # Stock level color
                if remaining < 5:
                    bg_color = (0.85, 0.20, 0.25, 1)
                elif remaining < 10:
                    bg_color = (0.95, 0.60, 0.10, 1)
                else:
                    try:
                        bg_color = tuple(float(x) for x in color.split(',')) if color else (0.2, 0.6, 0.86, 1)
                    except:
                        bg_color = (0.2, 0.6, 0.86, 1)

                # Main category button
                btn_text = f"{category}\nStock: {remaining}"
                if barcode:
                    btn_text += f"\n🔖 {barcode[:12]}"
                
                btn_cat = Button(
                    text=btn_text,
                    background_normal="",
                    background_color=bg_color,
                    color=(1, 1, 1, 1),
                    font_size=sp(13),
                )
                btn_cat.bind(
                    on_release=lambda inst, name=category: self.open_purchase_popup(name)
                )

                # Scan barcode button
                if BARCODE_AVAILABLE:
                    btn_scan = Button(
                        text="📷",
                        size_hint_x=None,
                        width=dp(45),
                        background_normal="",
                        background_color=(0.20, 0.60, 0.40, 1),
                        color=(1, 1, 1, 1),
                        font_size=sp(18),
                    )
                    btn_scan.bind(
                        on_release=lambda inst, name=category: self.scan_barcode_for_category(name)
                    )
                    row.add_widget(btn_scan)

                # Edit button with icon
                btn_edit = Button(
                    text="",
                    size_hint_x=None,
                    width=dp(45),
                    background_normal="icons/edit.png",
                    background_down="icons/edit.png",
                    background_color=(1, 0.6, 0.2, 1),
                )
                btn_edit.bind(
                    on_release=lambda inst, old=category: self.open_edit_category_popup(old)
                )

                # Delete button with icon
                btn_del = Button(
                    text="",
                    size_hint_x=None,
                    width=dp(45),
                    background_normal="icons/delete.png",
                    background_down="icons/delete.png",
                    background_color=(0.85, 0.20, 0.25, 1),
                )
                btn_del.bind(
                    on_release=lambda inst, name=category: self.delete_category_popup(name)
                )

                row.add_widget(btn_cat)
                row.add_widget(btn_edit)
                row.add_widget(btn_del)
                grid.add_widget(row)
        except Exception as e:
            print(f"Load categories error: {e}")

    def scan_barcode_for_category(self, category):
        """Scan and assign barcode to category"""
        def on_barcode_scanned(barcode):
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        "UPDATE inventory SET barcode=? WHERE category=?",
                        (barcode, category)
                    )
                    conn.commit()
                
                Popup(
                    title="✓ Barcode Saved",
                    content=Label(text=f"Barcode {barcode}\nassigned to {category}"),
                    size_hint=(0.7, 0.4)
                ).open()
                
                self.load_categories()
            except Exception as e:
                print(f"Save barcode error: {e}")
        
        BarcodeScannerPopup(callback=on_barcode_scanned).open()

    def add_category_popup(self):
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        
        ti_name = TextInput(hint_text="Category name", multiline=False)
        layout.add_widget(Label(text="Category Name", size_hint_y=None, height=20))
        layout.add_widget(ti_name)

        ti_barcode = TextInput(hint_text="Barcode (optional)", multiline=False)
        layout.add_widget(Label(text="Barcode", size_hint_y=None, height=20))
        layout.add_widget(ti_barcode)

        ti_color = TextInput(
            hint_text="Color (R,G,B,A)",
            text="0.2,0.6,0.86,1",
            multiline=False
        )
        layout.add_widget(Label(text="Button Color", size_hint_y=None, height=20))
        layout.add_widget(ti_color)

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_ok = Button(text="Add")
        btn_scan = Button(text="📷 Scan Barcode") if BARCODE_AVAILABLE else None
        btn_cancel = Button(text="Cancel")
        
        btn_row.add_widget(btn_ok)
        if btn_scan:
            btn_row.add_widget(btn_scan)
        btn_row.add_widget(btn_cancel)
        layout.add_widget(btn_row)

        popup = Popup(
            title="Add Category",
            content=layout,
            size_hint=(0.9, 0.6),
            auto_dismiss=False,
        )

        def do_add(_instance):
            name = ti_name.text.strip()
            barcode = ti_barcode.text.strip()
            color = ti_color.text.strip()
            if not name:
                return
            try:
                user = getattr(self.manager, 'admin_name', 'system')
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        """
                        INSERT OR IGNORE INTO inventory(
                            category, barcode, purchase_date, purchased_stock, 
                            cost_price, color, created_by
                        )
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        (name, barcode, "", 0, 0.0, color, user),
                    )
                    conn.commit()
                popup.dismiss()
                self.load_categories()
                self.load_low_stock()
                self.open_purchase_popup(name)
            except Exception as e:
                print(f"Add category error: {e}")
                popup.dismiss()

        def scan_and_fill(_instance):
            def on_scanned(barcode):
                ti_barcode.text = barcode
            BarcodeScannerPopup(callback=on_scanned).open()

        btn_ok.bind(on_release=do_add)
        if btn_scan:
            btn_scan.bind(on_release=scan_and_fill)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def open_edit_category_popup(self, old_name):
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        ti_name = TextInput(text=old_name, multiline=False)
        layout.add_widget(Label(text="Edit category name"))
        layout.add_widget(ti_name)

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_ok = Button(text="Save")
        btn_cancel = Button(text="Cancel")
        btn_row.add_widget(btn_ok)
        btn_row.add_widget(btn_cancel)
        layout.add_widget(btn_row)

        popup = Popup(
            title="Edit Category",
            content=layout,
            size_hint=(0.8, 0.4),
            auto_dismiss=False,
        )

        def do_save(_instance):
            new_name = ti_name.text.strip()
            if not new_name:
                return
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        "UPDATE inventory SET category=? WHERE category=?",
                        (new_name, old_name),
                    )
                    c.execute(
                        "UPDATE transactions SET category=? WHERE category=?",
                        (new_name, old_name),
                    )
                    conn.commit()
                popup.dismiss()
                self.load_categories()
            except Exception as e:
                print(f"Edit category error: {e}")
                popup.dismiss()

        btn_ok.bind(on_release=do_save)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def delete_category_popup(self, category):
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        layout.add_widget(
            Label(
                text=f"⚠ Delete '{category}' and all transactions?\n\nThis cannot be undone!",
                halign="center",
                valign="middle",
            )
        )

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_yes = Button(text="Yes, Delete", background_color=(1, 0.3, 0.3, 1))
        btn_no = Button(text="Cancel")
        btn_row.add_widget(btn_yes)
        btn_row.add_widget(btn_no)
        layout.add_widget(btn_row)

        popup = Popup(
            title="⚠ Confirm Delete",
            content=layout,
            size_hint=(0.8, 0.4),
            auto_dismiss=False,
        )

        def do_delete(_instance):
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute("DELETE FROM inventory WHERE category=?", (category,))
                    c.execute("DELETE FROM transactions WHERE category=?", (category,))
                    conn.commit()
                popup.dismiss()
                self.load_categories()
                self.load_low_stock()
            except Exception as e:
                print(f"Delete category error: {e}")
                popup.dismiss()

        btn_yes.bind(on_release=do_delete)
        btn_no.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def open_purchase_popup(self, category):
        self.selected_category = category

        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)

        name_input = TextInput(text=category, multiline=False, readonly=True)
        date_input = TextInput(
            hint_text="Purchase date (YYYY-MM-DD)",
            text=str(date.today()),
            multiline=False,
        )
        stock_input = TextInput(
            hint_text="Purchased stock (units)",
            multiline=False,
            input_filter="int",
        )
        price_input = TextInput(
            hint_text="Cost price per unit",
            multiline=False,
            input_filter="float",
        )
        expiry_input = TextInput(
            hint_text="Expiry date (YYYY-MM-DD, optional)",
            multiline=False,
        )

        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT purchase_date, purchased_stock, cost_price, expiry_date
                    FROM inventory WHERE category=?
                    """,
                    (category,),
                )
                row = c.fetchone()
                
                c.execute(
                    "SELECT MAX(purchase_date) FROM inventory WHERE category=? AND purchase_date != ''",
                    (category,)
                )
                last_restock = c.fetchone()[0]

            restock_label = Label(
                text=f"Last restocked: {last_restock or 'Never'}",
                size_hint_y=None,
                height=dp(20),
                font_size=sp(11),
                color=(0.7, 0.7, 0.7, 1)
            )

            if row:
                if row[0]:
                    date_input.text = row[0]
                if row[1] is not None:
                    stock_input.text = str(row[1])
                if row[2] is not None:
                    price_input.text = str(row[2])
                if row[3]:
                    expiry_input.text = row[3]
        except Exception as e:
            print(f"Open purchase popup error: {e}")
            restock_label = Label(text="Error loading data", size_hint_y=None, height=dp(20))

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)

        def save_and_close(_instance):
            self.save_purchase_row(
                category,
                date_input.text,
                stock_input.text,
                price_input.text,
                expiry_input.text
            )
            popup.dismiss()

        btn_save = Button(text="Save", on_release=save_and_close)
        btn_cancel = Button(text="Cancel", on_release=lambda *_: popup.dismiss())
        btn_row.add_widget(btn_save)
        btn_row.add_widget(btn_cancel)

        layout.add_widget(Label(text="Purchase / Restock", size_hint_y=None, height=20))
        layout.add_widget(restock_label)
        for w in (name_input, date_input, stock_input, price_input, expiry_input, btn_row):
            layout.add_widget(w)

        popup = Popup(
            title=f"Item: {category}",
            content=layout,
            size_hint=(0.9, 0.75),
            auto_dismiss=False,
        )
        popup.open()

    def save_purchase_row(self, category, purchase_date, stock_text, cost_price_text, expiry_date=""):
        def _to_int(v):
            try:
                return int(v)
            except:
                return 0

        def _to_float(v):
            try:
                return float(v)
            except:
                return 0.0

        purchased_stock = _to_int(stock_text)
        cost_price = _to_float(cost_price_text)

        if purchased_stock <= 0 or cost_price <= 0:
            Popup(
                title="Error",
                content=Label(text="Stock and price must be greater than 0"),
                size_hint=(0.6, 0.4),
            ).open()
            return

        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("SELECT id FROM inventory WHERE category=?", (category,))
                row = c.fetchone()

                if row:
                    c.execute(
                        """
                        UPDATE inventory
                        SET purchase_date=?, purchased_stock=?, cost_price=?, expiry_date=?
                        WHERE id=?
                        """,
                        (purchase_date, purchased_stock, cost_price, expiry_date, row[0]),
                    )
                else:
                    c.execute(
                        """
                        INSERT INTO inventory(
                            category, purchase_date, purchased_stock, cost_price, expiry_date
                        )
                        VALUES (?,?,?,?,?)
                        """,
                        (category, purchase_date, purchased_stock, cost_price, expiry_date),
                    )
                conn.commit()

            self.load_categories()
            self.load_low_stock()
            
            try:
                inv = self.manager.get_screen("inventory")
                inv.load_inventory()
            except:
                pass
        except Exception as e:
            print(f"Save purchase error: {e}")


class CategoryScreen(Screen):
    pass


# ========== DASHBOARD SCREEN (ENHANCED WITH CHARTS) ==========

class DashboardScreen(Screen):
    def on_enter(self):
        self.load_dashboard()
        if CHARTS_AVAILABLE:
            self.load_charts()

    def load_dashboard(self):
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                
                # Today's profit
                today = str(date.today())
                c.execute(
                    "SELECT COALESCE(SUM(profit), 0) FROM transactions WHERE date=?",
                    (today,)
                )
                today_profit = c.fetchone()[0] or 0
                
                # This week
                week_ago = str(date.today() - timedelta(days=7))
                c.execute(
                    "SELECT COALESCE(SUM(profit), 0) FROM transactions WHERE date >= ?",
                    (week_ago,)
                )
                week_profit = c.fetchone()[0] or 0
                
                # This month
                month_ago = str(date.today() - timedelta(days=30))
                c.execute(
                    "SELECT COALESCE(SUM(profit), 0) FROM transactions WHERE date >= ?",
                    (month_ago,)
                )
                month_profit = c.fetchone()[0] or 0
                
                # Best sellers
                c.execute(
                    """
                    SELECT category, SUM(quantity) as total_qty, SUM(profit) as total_profit
                    FROM transactions
                    WHERE date >= ?
                    GROUP BY category
                    ORDER BY total_qty DESC
                    LIMIT 5
                    """,
                    (month_ago,)
                )
                best_sellers = c.fetchall()
                
                # Total inventory value
                c.execute(
                    """
                    SELECT COALESCE(SUM(i.purchased_stock * i.cost_price), 0) - 
                           COALESCE(SUM(t.quantity * i.cost_price), 0)
                    FROM inventory i
                    LEFT JOIN transactions t ON i.category = t.category
                    """
                )
                inventory_value = c.fetchone()[0] or 0
                
                # Total customers
                c.execute("SELECT COUNT(*) FROM customers")
                total_customers = c.fetchone()[0] or 0

            # Update UI
            self.ids.lbl_today_profit.text = format_currency(today_profit)
            self.ids.lbl_week_profit.text = format_currency(week_profit)
            self.ids.lbl_month_profit.text = format_currency(month_profit)
            self.ids.lbl_inventory_value.text = format_currency(inventory_value)

            # Best sellers
            best_grid = self.ids.best_sellers_grid
            best_grid.clear_widgets()
            
            if not best_sellers:
                best_grid.add_widget(Label(
                    text="No sales data yet",
                    color=(0.7, 0.7, 0.7, 1),
                    size_hint_y=None,
                    height=dp(40)
                ))
            else:
                for rank, (category, qty, profit) in enumerate(best_sellers, 1):
                    row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(36), spacing=dp(4))
                    
                    # Medal emoji for top 3
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                    
                    row.add_widget(Label(
                        text=medal, 
                        size_hint_x=0.15, 
                        color=(1,1,1,1),
                        font_size=sp(16)
                    ))
                    row.add_widget(Label(
                        text=category, 
                        size_hint_x=0.4, 
                        color=(1,1,1,1),
                        halign="left",
                        text_size=(None, None)
                    ))
                    row.add_widget(Label(
                        text=f"{int(qty)} sold", 
                        size_hint_x=0.25, 
                        color=(0.7,0.9,1,1),
                        font_size=sp(12)
                    ))
                    row.add_widget(Label(
                        text=format_currency(profit), 
                        size_hint_x=0.2, 
                        color=(0.5,0.9,0.5,1),
                        bold=True
                    ))
                    best_grid.add_widget(row)
                    
                    # Add forecast if available
                    if FORECASTING_AVAILABLE and rank <= 3:
                        forecast = SalesForecaster.forecast_next_week(category)
                        if forecast:
                            trend_icon = "📈" if forecast['trend'] == 'up' else "📉"
                            forecast_row = BoxLayout(size_hint_y=None, height=dp(24), spacing=dp(4))
                            forecast_row.add_widget(Label(text="", size_hint_x=0.15))
                            forecast_row.add_widget(Label(
                                text=f"{trend_icon} Forecast: {forecast['next_week_total']} units next week",
                                size_hint_x=0.85,
                                color=(0.7, 0.7, 0.9, 1),
                                font_size=sp(10),
                                italic=True,
                                halign="left",
                                text_size=(None, None)
                            ))
                            best_grid.add_widget(forecast_row)
        except Exception as e:
            print(f"Load dashboard error: {e}")

    def load_charts(self):
        """Load profit chart"""
        try:
            chart_box = self.ids.chart_container
            chart_box.clear_widgets()
            
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                # Get last 30 days
                c.execute("""
                    SELECT date, COALESCE(SUM(profit), 0)
                    FROM transactions
                    WHERE date >= date('now', '-30 days')
                    GROUP BY date
                    ORDER BY date
                """)
                data = c.fetchall()
            
            if data and len(data) > 1:
                fig = ChartGenerator.generate_profit_chart(data)
                if fig:
                    chart_box.add_widget(FigureCanvasKivyAgg(fig))
            else:
                chart_box.add_widget(Label(
                    text="Not enough data for chart\n(Need at least 2 days of sales)",
                    color=(0.7, 0.7, 0.7, 1)
                ))
        except Exception as e:
            print(f"Chart loading error: {e}")


# ========== TRANSACTION SCREEN (ENHANCED) ==========

class TransactionScreen(Screen):
    current_category = StringProperty("")
    remaining_stock = NumericProperty(0)
    selected_customer_id = NumericProperty(0)

    def on_enter(self):
        self.refresh_category_buttons()
        self.load_recent_transactions()

    def refresh_category_buttons(self):
        try:
            grid = self.ids.trans_category_grid
            grid.clear_widgets()

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT i.category, i.purchased_stock, i.barcode,
                           COALESCE(SUM(t.quantity), 0) as sold
                    FROM inventory i
                    LEFT JOIN transactions t ON i.category = t.category
                    GROUP BY i.category
                    ORDER BY i.category
                    """
                )
                rows = c.fetchall()

            for category, purchased_stock, barcode, sold in rows:
                remaining = (purchased_stock or 0) - (sold or 0)
                
                btn_text = f"{category} ({remaining} left)"
                if barcode:
                    btn_text += f"\n🔖 {barcode[:15]}"
                
                btn = Button(
                    text=btn_text,
                    size_hint_y=None,
                    height=dp(50),
                    background_normal="",
                    background_color=(0.2, 0.6, 0.86, 1) if remaining > 0 else (0.5, 0.5, 0.5, 1),
                    color=(1, 1, 1, 1),
                    font_size=sp(13)
                )
                btn.bind(
                    on_release=lambda inst, name=category: self.set_current_category(name)
                )
                grid.add_widget(btn)
            
            # Add scan barcode button
            if BARCODE_AVAILABLE:
                scan_btn = Button(
                    text="📷 Scan Barcode to Sell",
                    size_hint_y=None,
                    height=dp(50),
                    background_normal="",
                    background_color=(0.13, 0.70, 0.35, 1),
                    color=(1, 1, 1, 1),
                    font_size=sp(14),
                    bold=True
                )
                scan_btn.bind(on_release=self.scan_to_sell)
                grid.add_widget(scan_btn)
        except Exception as e:
            print(f"Refresh category buttons error: {e}")

    def scan_to_sell(self, instance):
        """Scan barcode and auto-select category"""
        def on_scanned(barcode):
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT category FROM inventory WHERE barcode=?",
                        (barcode,)
                    )
                    result = c.fetchone()
                
                if result:
                    self.set_current_category(result[0])
                    # Auto-focus quantity input
                    self.ids.ti_qty.focus = True
                else:
                    Popup(
                        title="Not Found",
                        content=Label(text=f"No product found with barcode:\n{barcode}"),
                        size_hint=(0.7, 0.4)
                    ).open()
            except Exception as e:
                print(f"Scan to sell error: {e}")
        
        BarcodeScannerPopup(callback=on_scanned).open()

    def set_current_category(self, name):
        try:
            self.current_category = name

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT purchased_stock, cost_price
                    FROM inventory WHERE category=?
                    """,
                    (name,),
                )
                inv_row = c.fetchone()
                c.execute(
                    "SELECT COALESCE(SUM(quantity),0) FROM transactions WHERE category=?",
                    (name,),
                )
                sold_total = c.fetchone()[0] or 0

            if inv_row:
                purchased_stock, cost_price = inv_row
                self.remaining_stock = max((purchased_stock or 0) - sold_total, 0)
                self.ids.lbl_selected_item.text = (
                    f"📦 {name}\n"
                    f"Remaining: {self.remaining_stock} | Cost: {format_currency(cost_price)}"
                )
                # Suggested sell price (30% markup)
                suggested_price = (cost_price or 0) * 1.3
                self.ids.ti_sell_price.text = f"{suggested_price:.2f}"
            else:
                self.remaining_stock = 0
                self.ids.lbl_selected_item.text = "⚠ No purchase info for this item"
        except Exception as e:
            print(f"Set current category error: {e}")

    def select_customer(self):
        """Select customer for sale"""
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Search
        search = TextInput(hint_text="Search customer...", size_hint_y=None, height=dp(40))
        layout.add_widget(search)
        
        # Customer list
        scroll = ScrollView()
        customer_list = GridLayout(cols=1, size_hint_y=None, spacing=5)
        customer_list.bind(minimum_height=customer_list.setter('height'))
        
        def load_customers(search_text=""):
            customer_list.clear_widgets()
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        """
                        SELECT id, name, phone, loyalty_points
                        FROM customers
                        WHERE name LIKE ? OR phone LIKE ?
                        ORDER BY name
                        """,
                        (f"%{search_text}%", f"%{search_text}%")
                    )
                    customers = c.fetchall()
                
                for cust_id, name, phone, points in customers:
                    btn = Button(
                        text=f"{name} | {phone or 'No phone'} | 🎁 {points} pts",
                        size_hint_y=None,
                        height=dp(45)
                    )
                    btn.bind(on_release=lambda inst, cid=cust_id, cname=name: select_this_customer(cid, cname))
                    customer_list.add_widget(btn)
                
                # Add new customer button
                add_btn = Button(
                    text="➕ Add New Customer",
                    size_hint_y=None,
                    height=dp(50),
                    background_color=(0.13, 0.70, 0.35, 1)
                )
                add_btn.bind(on_release=lambda inst: add_new_customer())
                customer_list.add_widget(add_btn)
            except Exception as e:
                print(f"Load customers error: {e}")
        
        def select_this_customer(cust_id, cust_name):
            self.selected_customer_id = cust_id
            self.ids.lbl_customer.text = f"👤 Customer: {cust_name}"
            popup.dismiss()
        
        def add_new_customer():
            popup.dismiss()
            self.manager.current = "customers"
        
        search.bind(text=lambda inst, val: load_customers(val))
        load_customers()
        
        scroll.add_widget(customer_list)
        layout.add_widget(scroll)
        
        popup = Popup(
            title="Select Customer",
            content=layout,
            size_hint=(0.9, 0.8)
        )
        popup.open()

    def add_transaction(self, quantity_text, sell_price_text):
        if not self.current_category:
            Popup(
                title="Error",
                content=Label(text="Select an item first"),
                size_hint=(0.6, 0.4),
            ).open()
            return

        try:
            qty = int(quantity_text)
        except:
            qty = 0
        try:
            sell_price = float(sell_price_text)
        except:
            sell_price = 0.0

        if qty <= 0:
            Popup(
                title="Error",
                content=Label(text="Quantity must be > 0"),
                size_hint=(0.6, 0.4),
            ).open()
            return

        if sell_price <= 0:
            Popup(
                title="Error",
                content=Label(text="Sell price must be > 0"),
                size_hint=(0.6, 0.4),
            ).open()
            return

        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT purchased_stock, cost_price FROM inventory WHERE category=?",
                    (self.current_category,),
                )
                inv_row = c.fetchone()
                if not inv_row:
                    Popup(
                        title="Error",
                        content=Label(text="No purchase record for this item"),
                        size_hint=(0.6, 0.4),
                    ).open()
                    return

                purchased_stock, cost_price = inv_row
                c.execute(
                    "SELECT COALESCE(SUM(quantity),0) FROM transactions WHERE category=?",
                    (self.current_category,),
                )
                sold_total = c.fetchone()[0] or 0
                remaining = (purchased_stock or 0) - sold_total

                if qty > remaining:
                    Popup(
                        title="Insufficient Stock",
                        content=Label(text=f"Not enough stock!\n\nAvailable: {remaining} units\nRequested: {qty} units"),
                        size_hint=(0.7, 0.4),
                    ).open()
                    return

                # Check for active promotions
                discount = 0
                c.execute(
                    """
                    SELECT discount_percent, discount_amount
                    FROM promotions
                    WHERE category=? AND active=1
                    AND date('now') BETWEEN start_date AND end_date
                    LIMIT 1
                    """,
                    (self.current_category,)
                )
                promo = c.fetchone()
                if promo:
                    if promo[0]:  # Percentage discount
                        discount = (sell_price * qty) * (promo[0] / 100)
                    elif promo[1]:  # Fixed discount
                        discount = promo[1]

                # Warn if selling below cost
                if sell_price < (cost_price or 0):
                    layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
                    layout.add_widget(Label(
                        text=f"⚠ Warning!\n\nSelling below cost!\n\nCost: {format_currency(cost_price)}\nSell: {format_currency(sell_price)}\n\nContinue?",
                        halign="center"
                    ))
                    btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
                    btn_yes = Button(text="Yes, Continue")
                    btn_no = Button(text="Cancel")
                    btn_row.add_widget(btn_yes)
                    btn_row.add_widget(btn_no)
                    layout.add_widget(btn_row)
                    
                    popup = Popup(title="⚠ Low Price Warning", content=layout, size_hint=(0.8, 0.5))
                    btn_yes.bind(on_release=lambda *_: self._do_transaction(qty, sell_price, cost_price, discount, popup))
                    btn_no.bind(on_release=lambda *_: popup.dismiss())
                    popup.open()
                    return

                self._do_transaction(qty, sell_price, cost_price, discount, None)
        except Exception as e:
            print(f"Add transaction error: {e}")
            Popup(
                title="Error",
                content=Label(text=f"Transaction failed:\n{str(e)}"),
                size_hint=(0.7, 0.4),
            ).open()

    def _do_transaction(self, qty, sell_price, cost_price, discount, warning_popup):
        if warning_popup:
            warning_popup.dismiss()

        try:
            profit_per_unit = sell_price - (cost_price or 0.0)
            profit = (profit_per_unit * qty) - discount
            profit_margin = (profit / (sell_price * qty) * 100) if sell_price > 0 else 0
            
            user = getattr(self.manager, 'admin_name', 'system')
            current_time = datetime.now().strftime("%H:%M:%S")

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO transactions(
                        type, category, amount, date, time,
                        quantity, sell_price, profit, discount,
                        customer_id, user_id
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "sale",
                        self.current_category,
                        sell_price * qty,
                        datetime.now().strftime("%Y-%m-%d"),
                        current_time,
                        qty,
                        sell_price,
                        profit,
                        discount,
                        self.selected_customer_id if self.selected_customer_id > 0 else None,
                        user
                    ),
                )
                trans_id = c.lastrowid
                
                # Update customer loyalty points
                if self.selected_customer_id > 0:
                    points = int(profit / 10)  # 1 point per ₱10 profit
                    c.execute(
                        """
                        UPDATE customers
                        SET loyalty_points = loyalty_points + ?,
                            total_purchases = total_purchases + ?,
                            last_purchase_date = ?
                        WHERE id = ?
                        """,
                        (points, sell_price * qty, datetime.now().strftime("%Y-%m-%d"), self.selected_customer_id)
                    )
                
                conn.commit()

            log_action(
                "SALE",
                f"{self.current_category}: qty={qty}, sell={sell_price}, profit={profit}",
                user
            )

            # Generate receipt
            if PDF_AVAILABLE:
                receipt_data = {
                    'id': trans_id,
                    'category': self.current_category,
                    'quantity': qty,
                    'sell_price': sell_price,
                    'amount': sell_price * qty,
                    'discount': discount,
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'time': current_time,
                    'payment_method': self.ids.payment_method.text if hasattr(self.ids, 'payment_method') else 'Cash'
                }
                receipt_file = ReceiptGenerator.generate(receipt_data)

            # Play success sound
            safe_play_sound('assets/sale.wav')

            # Show success popup
            layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
            layout.add_widget(Label(
                text=f"✓ Sale Recorded!\n\nProfit: {format_currency(profit)}\nMargin: {profit_margin:.1f}%",
                font_size=sp(16),
                halign="center"
            ))
            if discount > 0:
                layout.add_widget(Label(
                    text=f"💰 Discount applied: {format_currency(discount)}",
                    font_size=sp(13),
                    color=(1, 0.8, 0.2, 1)
                ))
            if PDF_AVAILABLE and receipt_file:
                btn_receipt = Button(
                    text="📄 View Receipt",
                    size_hint_y=None,
                    height=dp(40)
                )
                btn_receipt.bind(on_release=lambda inst: self.open_receipt(receipt_file))
                layout.add_widget(btn_receipt)
            
            Popup(
                title="✓ Success",
                content=layout,
                size_hint=(0.7, 0.4),
            ).open()

            self.set_current_category(self.current_category)
            
            try:
                inv = self.manager.get_screen("inventory")
                inv.load_inventory()
            except:
                pass
            
            try:
                home = self.manager.get_screen("home")
                home.load_low_stock()
            except:
                pass

            self.ids.ti_qty.text = ""
            self.ids.ti_sell_price.text = ""
            self.selected_customer_id = 0
            self.ids.lbl_customer.text = "👤 No customer selected"
            
            self.load_recent_transactions()
        except Exception as e:
            print(f"Do transaction error: {e}")
            Popup(
                title="Error",
                content=Label(text=f"Failed to complete sale:\n{str(e)}"),
                size_hint=(0.7, 0.4)
            ).open()

    def open_receipt(self, filepath):
        """Open/share receipt file"""
        try:
            import subprocess
            subprocess.Popen(['xdg-open', filepath])
        except:
            Popup(
                title="Receipt Saved",
                content=Label(text=f"Receipt saved to:\n{filepath}"),
                size_hint=(0.8, 0.4)
            ).open()

    def load_recent_transactions(self):
        """Load last 5 transactions with undo"""
        try:
            grid = self.ids.recent_trans_grid
            grid.clear_widgets()

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT t.id, t.category, t.quantity, t.sell_price, 
                           t.profit, t.date, t.time, c.name
                    FROM transactions t
                    LEFT JOIN customers c ON t.customer_id = c.id
                    ORDER BY t.id DESC
                    LIMIT 5
                    """
                )
                rows = c.fetchall()

            if not rows:
                grid.add_widget(Label(
                    text="No recent transactions",
                    color=(0.7, 0.7, 0.7, 1),
                    size_hint_y=None,
                    height=dp(30)
                ))
                return

            for trans_id, category, qty, price, profit, trans_date, trans_time, customer_name in rows:
                row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(45), spacing=dp(4))
                
                customer_text = f" | {customer_name}" if customer_name else ""
                info_label = Label(
                    text=f"{category}: {qty}x{format_currency(price)}\n= {format_currency(profit)} @ {trans_time or ''}",
                    size_hint_x=0.65,
                    color=(1, 1, 1, 1),
                    font_size=sp(11),
                    halign="left",
                    valign="middle",
                    text_size=(None, None)
                )
                
                undo_btn = Button(
                    text="↩ Undo",
                    size_hint_x=0.35,
                    background_normal="",
                    background_color=(0.85, 0.20, 0.25, 1),
                    color=(1, 1, 1, 1),
                    font_size=sp(12)
                )
                undo_btn.bind(on_release=lambda inst, tid=trans_id: self.undo_transaction(tid))
                
                row.add_widget(info_label)
                row.add_widget(undo_btn)
                grid.add_widget(row)
        except Exception as e:
            print(f"Load recent transactions error: {e}")

    def undo_transaction(self, trans_id):
        """Delete transaction (undo)"""
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        layout.add_widget(Label(text="⚠ Undo this sale?\n\nThis will restore stock and remove from records.", halign="center"))

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_yes = Button(text="Yes, Undo")
        btn_no = Button(text="Cancel")
        btn_row.add_widget(btn_yes)
        btn_row.add_widget(btn_no)
        layout.add_widget(btn_row)

        popup = Popup(title="Confirm Undo", content=layout, size_hint=(0.7, 0.4))

        def do_undo(_):
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    
                    # Get transaction details
                    c.execute("SELECT customer_id, profit FROM transactions WHERE id=?", (trans_id,))
                    trans = c.fetchone()
                    
                    if trans and trans[0]:  # Reverse loyalty points
                        points = int(trans[1] / 10)
                        c.execute(
                            "UPDATE customers SET loyalty_points = loyalty_points - ? WHERE id = ?",
                            (points, trans[0])
                        )
                    
                    c.execute("DELETE FROM transactions WHERE id=?", (trans_id,))
                    conn.commit()
                
                popup.dismiss()
                self.load_recent_transactions()
                self.refresh_category_buttons()
                
                try:
                    inv = self.manager.get_screen("inventory")
                    inv.load_inventory()
                except:
                    pass
                
                try:
                    home = self.manager.get_screen("home")
                    home.load_low_stock()
                except:
                    pass
            except Exception as e:
                print(f"Undo transaction error: {e}")
                popup.dismiss()

        btn_yes.bind(on_release=do_undo)
        btn_no.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

# ========== CUSTOMERS SCREEN ==========

class CustomersScreen(Screen):
    def on_enter(self):
        self.load_customers()

    def load_customers(self):
        try:
            grid = self.ids.customers_grid
            grid.clear_widgets()

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT id, name, phone, email, loyalty_points, 
                           total_purchases, last_purchase_date
                    FROM customers
                    ORDER BY total_purchases DESC
                    """
                )
                rows = c.fetchall()

            if not rows:
                grid.add_widget(Label(
                    text="No customers yet.\nTap '+' to add your first customer!",
                    color=(0.7, 0.7, 0.7, 1),
                    size_hint_y=None,
                    height=dp(60)
                ))
                return

            for cust_id, name, phone, email, points, purchases, last_date in rows:
                card = BoxLayout(
                    orientation='vertical',
                    size_hint_y=None,
                    height=dp(110),
                    spacing=dp(6),
                    padding=dp(12)
                )
                card.canvas.before.clear()
                with card.canvas.before:
                    Color(0.12, 0.12, 0.15, 1)
                    RoundedRectangle(pos=card.pos, size=card.size, radius=[dp(12)])
                card.bind(pos=lambda inst, val, c=card: setattr(c.canvas.before.children[-1], 'pos', val))
                card.bind(size=lambda inst, val, c=card: setattr(c.canvas.before.children[-1], 'size', val))

                # Name
                name_label = Label(
                    text=f"👤 {name}",
                    font_size=sp(16),
                    bold=True,
                    color=(1, 1, 1, 1),
                    size_hint_y=None,
                    height=dp(25),
                    halign="left",
                    valign="middle",
                    text_size=(dp(250), None)
                )
                card.add_widget(name_label)

                # Contact info
                contact_label = Label(
                    text=f"📞 {phone or 'No phone'} | 📧 {email or 'No email'}",
                    font_size=sp(11),
                    color=(0.8, 0.8, 0.8, 1),
                    size_hint_y=None,
                    height=dp(18),
                    halign="left",
                    valign="middle",
                    text_size=(dp(250), None)
                )
                card.add_widget(contact_label)

                # Stats
                stats_label = Label(
                    text=f"🎁 {points} points | 💰 {format_currency(purchases or 0)} spent | Last: {last_date or 'Never'}",
                    font_size=sp(10),
                    color=(0.5, 0.9, 0.5, 1),
                    size_hint_y=None,
                    height=dp(18),
                    halign="left",
                    valign="middle",
                    text_size=(dp(250), None)
                )
                card.add_widget(stats_label)

                # Action buttons
                btn_row = BoxLayout(size_hint_y=None, height=dp(35), spacing=dp(6))
                
                btn_edit = Button(
                    text="✎ Edit",
                    background_normal="",
                    background_color=(0.35, 0.35, 0.40, 1),
                    font_size=sp(12)
                )
                btn_edit.bind(on_release=lambda inst, cid=cust_id: self.edit_customer(cid))
                
                btn_history = Button(
                    text="📋 History",
                    background_normal="",
                    background_color=(0.2, 0.6, 0.86, 1),
                    font_size=sp(12)
                )
                btn_history.bind(on_release=lambda inst, cid=cust_id: self.show_purchase_history(cid))
                
                btn_delete = Button(
                    text="🗑",
                    size_hint_x=None,
                    width=dp(50),
                    background_normal="",
                    background_color=(0.85, 0.20, 0.25, 1),
                    font_size=sp(16)
                )
                btn_delete.bind(on_release=lambda inst, cid=cust_id: self.delete_customer(cid))
                
                btn_row.add_widget(btn_edit)
                btn_row.add_widget(btn_history)
                btn_row.add_widget(btn_delete)
                card.add_widget(btn_row)

                grid.add_widget(card)
        except Exception as e:
            print(f"Load customers error: {e}")

    def add_customer_popup(self):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        ti_name = TextInput(hint_text="Customer name *", multiline=False)
        ti_phone = TextInput(hint_text="Phone number", multiline=False)
        ti_email = TextInput(hint_text="Email", multiline=False)
        ti_address = TextInput(hint_text="Address", multiline=True, size_hint_y=None, height=dp(60))

        for label, widget in [
            ("Name *", ti_name),
            ("Phone", ti_phone),
            ("Email", ti_email),
            ("Address", ti_address)
        ]:
            layout.add_widget(Label(text=label, size_hint_y=None, height=20, halign="left", text_size=(None, None)))
            layout.add_widget(widget)

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_save = Button(text="💾 Save")
        btn_cancel = Button(text="Cancel")
        btn_row.add_widget(btn_save)
        btn_row.add_widget(btn_cancel)
        layout.add_widget(btn_row)

        popup = Popup(
            title="➕ Add New Customer",
            content=layout,
            size_hint=(0.9, 0.75),
            auto_dismiss=False
        )

        def do_save(_):
            name = ti_name.text.strip()
            if not name:
                Popup(
                    title="Error",
                    content=Label(text="Customer name is required"),
                    size_hint=(0.6, 0.3)
                ).open()
                return
            
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        """
                        INSERT INTO customers(name, phone, email, address, created_date)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (name, ti_phone.text.strip(), ti_email.text.strip(), 
                         ti_address.text.strip(), datetime.now().strftime("%Y-%m-%d"))
                    )
                    conn.commit()
                
                popup.dismiss()
                self.load_customers()
                
                Popup(
                    title="✓ Success",
                    content=Label(text=f"Customer '{name}' added!"),
                    size_hint=(0.6, 0.3)
                ).open()
            except Exception as e:
                print(f"Add customer error: {e}")
                Popup(
                    title="Error",
                    content=Label(text=f"Failed to add customer:\n{str(e)}"),
                    size_hint=(0.7, 0.4)
                ).open()

        btn_save.bind(on_release=do_save)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def edit_customer(self, customer_id):
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT name, phone, email, address FROM customers WHERE id=?",
                    (customer_id,)
                )
                row = c.fetchone()
            
            if not row:
                return

            layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

            ti_name = TextInput(text=row[0] or "", multiline=False)
            ti_phone = TextInput(text=row[1] or "", multiline=False)
            ti_email = TextInput(text=row[2] or "", multiline=False)
            ti_address = TextInput(text=row[3] or "", multiline=True, size_hint_y=None, height=dp(60))

            for label, widget in [
                ("Name", ti_name),
                ("Phone", ti_phone),
                ("Email", ti_email),
                ("Address", ti_address)
            ]:
                layout.add_widget(Label(text=label, size_hint_y=None, height=20))
                layout.add_widget(widget)

            btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
            btn_save = Button(text="💾 Save")
            btn_cancel = Button(text="Cancel")
            btn_row.add_widget(btn_save)
            btn_row.add_widget(btn_cancel)
            layout.add_widget(btn_row)

            popup = Popup(
                title="✎ Edit Customer",
                content=layout,
                size_hint=(0.9, 0.75),
                auto_dismiss=False
            )

            def do_save(_):
                try:
                    with sqlite3.connect(DB_NAME) as conn:
                        c = conn.cursor()
                        c.execute(
                            """
                            UPDATE customers
                            SET name=?, phone=?, email=?, address=?
                            WHERE id=?
                            """,
                            (ti_name.text.strip(), ti_phone.text.strip(), 
                             ti_email.text.strip(), ti_address.text.strip(), customer_id)
                        )
                        conn.commit()
                    popup.dismiss()
                    self.load_customers()
                except Exception as e:
                    print(f"Edit customer error: {e}")

            btn_save.bind(on_release=do_save)
            btn_cancel.bind(on_release=lambda *_: popup.dismiss())
            popup.open()
        except Exception as e:
            print(f"Edit customer error: {e}")

    def delete_customer(self, customer_id):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        layout.add_widget(Label(
            text="⚠ Delete this customer?\n\nPurchase history will remain but\nwill not be linked to any customer.",
            halign="center"
        ))

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_yes = Button(text="Yes, Delete", background_color=(1, 0.3, 0.3, 1))
        btn_no = Button(text="Cancel")
        btn_row.add_widget(btn_yes)
        btn_row.add_widget(btn_no)
        layout.add_widget(btn_row)

        popup = Popup(title="Confirm Delete", content=layout, size_hint=(0.75, 0.4))

        def do_delete(_):
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute("DELETE FROM customers WHERE id=?", (customer_id,))
                    conn.commit()
                popup.dismiss()
                self.load_customers()
            except Exception as e:
                print(f"Delete customer error: {e}")

        btn_yes.bind(on_release=do_delete)
        btn_no.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def show_purchase_history(self, customer_id):
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("SELECT name FROM customers WHERE id=?", (customer_id,))
                customer_name = c.fetchone()[0]
                
                c.execute(
                    """
                    SELECT date, time, category, quantity, amount, profit
                    FROM transactions
                    WHERE customer_id=?
                    ORDER BY date DESC, time DESC
                    LIMIT 50
                    """,
                    (customer_id,)
                )
                history = c.fetchall()

            layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
            
            # Summary
            if history:
                total_spent = sum(h[4] for h in history)
                total_items = sum(h[3] for h in history)
                summary = Label(
                    text=f"Total: {format_currency(total_spent)} | {total_items} items | {len(history)} transactions",
                    size_hint_y=None,
                    height=dp(30),
                    color=(0.5, 0.9, 0.5, 1),
                    font_size=sp(13)
                )
                layout.add_widget(summary)

            scroll = ScrollView()
            history_grid = GridLayout(cols=1, size_hint_y=None, spacing=5)
            history_grid.bind(minimum_height=history_grid.setter('height'))

            if not history:
                history_grid.add_widget(Label(
                    text="No purchase history",
                    size_hint_y=None,
                    height=dp(40)
                ))
            else:
                for date, time, category, qty, amount, profit in history:
                    row = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40), spacing=5)
                    row.add_widget(Label(
                        text=f"{date}\n{time or ''}",
                        size_hint_x=0.3,
                        font_size=sp(10)
                    ))
                    row.add_widget(Label(
                        text=f"{category}",
                        size_hint_x=0.3,
                        font_size=sp(11)
                    ))
                    row.add_widget(Label(
                        text=f"{qty}x",
                        size_hint_x=0.15,
                        font_size=sp(11)
                    ))
                    row.add_widget(Label(
                        text=format_currency(amount),
                        size_hint_x=0.25,
                        font_size=sp(11),
                        color=(0.5, 0.9, 0.5, 1)
                    ))
                    history_grid.add_widget(row)

            scroll.add_widget(history_grid)
            layout.add_widget(scroll)

            Popup(
                title=f"📋 {customer_name}'s Purchase History",
                content=layout,
                size_hint=(0.9, 0.8)
            ).open()
        except Exception as e:
            print(f"Show history error: {e}")


# ========== INVENTORY SCREEN ==========

class InventoryScreen(Screen):
    def on_enter(self):
        try:
            if "ti_filter_date" in self.ids:
                self.ids.ti_filter_date.text = str(date.today())
            self.load_inventory()
            self.load_daily_summary(self.ids.ti_filter_date.text)
        except Exception as e:
            print(f"Inventory on_enter error: {e}")

    def load_inventory(self):
        try:
            table = self.ids.inventory_table
            # Keep header
            header = list(table.children[-9:])
            table.clear_widgets()
            for w in reversed(header):
                table.add_widget(w)

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT category, purchased_stock, cost_price
                    FROM inventory
                    ORDER BY category
                    """
                )
                inv_rows = c.fetchall()

                c2 = conn.cursor()
                for category, purchased_stock, cost_price in inv_rows:
                    purchased_stock = purchased_stock or 0
                    cost_price = cost_price or 0.0

                    c2.execute(
                        """
                        SELECT COALESCE(SUM(quantity),0),
                               COALESCE(SUM(quantity*sell_price),0),
                               COALESCE(SUM(profit),0),
                               COALESCE(AVG(sell_price),0)
                        FROM transactions
                        WHERE category=?
                        """,
                        (category,),
                    )
                    sold_qty, sales_total, total_profit, avg_sell_price = c2.fetchone()
                    sold_qty = sold_qty or 0
                    sales_total = sales_total or 0.0
                    total_profit = total_profit or 0.0
                    avg_sell_price = avg_sell_price or 0.0

                    remaining = max(purchased_stock - sold_qty, 0)
                    purchase_total = purchased_stock * cost_price
                    
                    if sales_total > 0:
                        profit_margin = (total_profit / sales_total) * 100
                        profit_display = f"{total_profit:.2f}\n({profit_margin:.0f}%)"
                    else:
                        profit_display = f"{total_profit:.2f}"

                    row_values = [
                        category or "",
                        str(purchased_stock),
                        str(remaining),
                        f"{cost_price:.2f}",
                        f"{purchase_total:.2f}",
                        str(sold_qty),
                        f"{avg_sell_price:.2f}",
                        f"{sales_total:.2f}",
                        profit_display,
                    ]

                    widths = [110, 90, 90, 80, 90, 70, 80, 100, 80]
                    for i, value in enumerate(row_values):
                        if i == 8 and sales_total > 0:
                            margin = (total_profit / sales_total) * 100
                            if margin > 20:
                                color = (0.5, 0.9, 0.5, 1)
                            elif margin > 10:
                                color = (1, 0.8, 0.2, 1)
                            else:
                                color = (1, 0.3, 0.3, 1)
                        else:
                            color = (1, 1, 1, 1)

                        table.add_widget(
                            Label(
                                text=value,
                                size_hint_x=None,
                                width=dp(widths[i]),
                                color=color,
                                font_size=12,
                            )
                        )
        except Exception as e:
            print(f"Load inventory error: {e}")

    def load_daily_summary(self, date_str):
        try:
            if not date_str.strip():
                date_str = str(date.today())
                if "ti_filter_date" in self.ids:
                    self.ids.ti_filter_date.text = date_str

            daily_table = self.ids.daily_table
            header = list(daily_table.children[-4:])
            daily_table.clear_widgets()
            for w in reversed(header):
                daily_table.add_widget(w)

            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT category, SUM(quantity), SUM(amount), SUM(profit)
                    FROM transactions
                    WHERE date=?
                    GROUP BY category
                    ORDER BY SUM(profit) DESC
                    """,
                    (date_str,),
                )
                rows = c.fetchall()

            if not rows:
                for _ in range(4):
                    daily_table.add_widget(Label(text="No sales", color=(0.7, 0.7, 0.7, 1)))
            else:
                for category, qty, sales, profit in rows:
                    daily_table.add_widget(Label(text=category or "", font_size=12))
                    daily_table.add_widget(Label(text=str(int(qty or 0)), font_size=12))
                    daily_table.add_widget(Label(text=f"{sales:.2f}", font_size=12, color=(0.7, 0.9, 1, 1)))
                    daily_table.add_widget(Label(text=f"{profit:.2f}", font_size=12, color=(0.5, 0.9, 0.5, 1)))
        except Exception as e:
            print(f"Load daily summary error: {e}")

    def quick_date(self, days_ago):
        """Set date to X days ago"""
        target_date = date.today() - timedelta(days=days_ago)
        self.ids.ti_filter_date.text = str(target_date)
        self.load_daily_summary(str(target_date))


# ========== ADMIN SCREEN ==========

class AdminScreen(Screen):
    def on_enter(self):
        self.load_settings()

    def load_settings(self):
        """Load current settings"""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("SELECT value FROM settings WHERE key='theme'")
                theme = c.fetchone()
                if theme:
                    self.ids.theme_spinner.text = theme[0].capitalize()
                
                c.execute("SELECT value FROM settings WHERE key='store_name'")
                store = c.fetchone()
                if store and hasattr(self.ids, 'store_name_input'):
                    self.ids.store_name_input.text = store[0]
        except Exception as e:
            print(f"Load settings error: {e}")

    def change_password(self):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        ti_old = TextInput(hint_text="Current password", password=True, multiline=False)
        ti_new = TextInput(hint_text="New password (min 8 chars)", password=True, multiline=False)
        ti_confirm = TextInput(hint_text="Confirm new password", password=True, multiline=False)

        for label, widget in [
            ("Current Password", ti_old),
            ("New Password", ti_new),
            ("Confirm Password", ti_confirm)
        ]:
            layout.add_widget(Label(text=label, size_hint_y=None, height=20))
            layout.add_widget(widget)

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_save = Button(text="💾 Change Password")
        btn_cancel = Button(text="Cancel")
        btn_row.add_widget(btn_save)
        btn_row.add_widget(btn_cancel)
        layout.add_widget(btn_row)

        popup = Popup(
            title="🔑 Change Password",
            content=layout,
            size_hint=(0.9, 0.6),
            auto_dismiss=False
        )

        def do_change(_):
            old_pw = ti_old.text
            new_pw = ti_new.text
            confirm_pw = ti_confirm.text

            if not all([old_pw, new_pw, confirm_pw]):
                Popup(
                    title="Error",
                    content=Label(text="All fields are required"),
                    size_hint=(0.6, 0.3)
                ).open()
                return

            if len(new_pw) < 8:
                Popup(
                    title="Error",
                    content=Label(text="New password must be at least 8 characters"),
                    size_hint=(0.7, 0.3)
                ).open()
                return

            if new_pw != confirm_pw:
                Popup(
                    title="Error",
                    content=Label(text="New passwords don't match"),
                    size_hint=(0.6, 0.3)
                ).open()
                return

            try:
                username = getattr(self.manager, 'admin_name', '')
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT id FROM users WHERE username=? AND password=?",
                        (username, old_pw)
                    )
                    if not c.fetchone():
                        Popup(
                            title="Error",
                            content=Label(text="Current password is incorrect"),
                            size_hint=(0.7, 0.3)
                        ).open()
                        return

                    c.execute(
                        "UPDATE users SET password=? WHERE username=?",
                        (new_pw, username)
                    )
                    conn.commit()

                popup.dismiss()
                log_action("PASSWORD_CHANGE", f"Password changed for {username}", username)
                
                Popup(
                    title="✓ Success",
                    content=Label(text="Password changed successfully!"),
                    size_hint=(0.6, 0.3)
                ).open()
            except Exception as e:
                print(f"Change password error: {e}")
                Popup(
                    title="Error",
                    content=Label(text=f"Failed to change password:\n{str(e)}"),
                    size_hint=(0.7, 0.4)
                ).open()

        btn_save.bind(on_release=do_change)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def export_data(self):
        """Export all data to CSV"""
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        layout.add_widget(Label(
            text="Export data to CSV files?\n\nFiles will be saved to:\n/storage/emulated/0/BookKeep/exports/",
            halign="center"
        ))

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_yes = Button(text="📥 Export", background_color=(0.13, 0.70, 0.35, 1))
        btn_no = Button(text="Cancel")
        btn_row.add_widget(btn_yes)
        btn_row.add_widget(btn_no)
        layout.add_widget(btn_row)

        popup = Popup(title="💾 Export Data", content=layout, size_hint=(0.85, 0.5))

        def do_export(_):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Export inventory
                with sqlite3.connect(DB_NAME) as conn:
                    df = conn.execute("SELECT * FROM inventory").fetchall()
                    columns = [desc[0] for desc in conn.execute("SELECT * FROM inventory").description]
                
                inv_file = os.path.join(EXPORT_DIR, f"inventory_{timestamp}.csv")
                with open(inv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(df)

                # Export transactions
                with sqlite3.connect(DB_NAME) as conn:
                    df = conn.execute("SELECT * FROM transactions").fetchall()
                    columns = [desc[0] for desc in conn.execute("SELECT * FROM transactions").description]
                
                trans_file = os.path.join(EXPORT_DIR, f"transactions_{timestamp}.csv")
                with open(trans_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(df)

                # Export customers
                with sqlite3.connect(DB_NAME) as conn:
                    df = conn.execute("SELECT * FROM customers").fetchall()
                    columns = [desc[0] for desc in conn.execute("SELECT * FROM customers").description]
                
                cust_file = os.path.join(EXPORT_DIR, f"customers_{timestamp}.csv")
                with open(cust_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(df)

                popup.dismiss()
                log_action("EXPORT", f"Data exported to {EXPORT_DIR}")
                
                Popup(
                    title="✓ Export Complete",
                    content=Label(
                        text=f"Data exported successfully!\n\n3 files saved to:\n{EXPORT_DIR}",
                        halign="center"
                    ),
                    size_hint=(0.85, 0.5)
                ).open()
            except Exception as e:
                print(f"Export error: {e}")
                popup.dismiss()
                Popup(
                    title="Error",
                    content=Label(text=f"Export failed:\n{str(e)}"),
                    size_hint=(0.7, 0.4)
                ).open()

        btn_yes.bind(on_release=do_export)
        btn_no.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def backup_database(self):
        """Manual database backup"""
        try:
            if auto_backup():
                Popup(
                    title="✓ Backup Complete",
                    content=Label(
                        text=f"Database backed up successfully!\n\nLocation:\n{BACKUP_DIR}",
                        halign="center"
                    ),
                    size_hint=(0.8, 0.4)
                ).open()
            else:
                Popup(
                    title="Error",
                    content=Label(text="Backup failed. Check permissions."),
                    size_hint=(0.7, 0.3)
                ).open()
        except Exception as e:
            print(f"Backup error: {e}")

    def view_audit_log(self):
        """View audit trail"""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT user, action, details, timestamp
                    FROM audit_log
                    ORDER BY id DESC
                    LIMIT 100
                    """
                )
                logs = c.fetchall()

            layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
            
            scroll = ScrollView()
            log_grid = GridLayout(cols=1, size_hint_y=None, spacing=3)
            log_grid.bind(minimum_height=log_grid.setter('height'))

            if not logs:
                log_grid.add_widget(Label(
                    text="No audit logs",
                    size_hint_y=None,
                    height=dp(40)
                ))
            else:
                for user, action, details, timestamp in logs:
                    log_text = f"[{timestamp}] {user}: {action}\n{details}"
                    log_grid.add_widget(Label(
                        text=log_text,
                        size_hint_y=None,
                        height=dp(50),
                        font_size=sp(10),
                        halign="left",
                        valign="top",
                        text_size=(dp(300), None)
                    ))

            scroll.add_widget(log_grid)
            layout.add_widget(scroll)

            Popup(
                title="📋 Audit Log (Last 100)",
                content=layout,
                size_hint=(0.95, 0.85)
            ).open()
        except Exception as e:
            print(f"View audit log error: {e}")

    def delete_account(self):
        """Delete admin account"""
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        layout.add_widget(Label(
            text="⚠⚠⚠ DANGER ZONE ⚠⚠⚠\n\nDelete your admin account?\n\nThis will:\n• Delete your user account\n• Keep all inventory & sales data\n• Allow new registration\n\nType your password to confirm:",
            halign="center",
            color=(1, 0.3, 0.3, 1)
        ))

        ti_password = TextInput(hint_text="Enter password", password=True, multiline=False, size_hint_y=None, height=dp(40))
        layout.add_widget(ti_password)

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_delete = Button(text="🗑 Delete Account", background_color=(1, 0.2, 0.2, 1))
        btn_cancel = Button(text="Cancel")
        btn_row.add_widget(btn_delete)
        btn_row.add_widget(btn_cancel)
        layout.add_widget(btn_row)

        popup = Popup(
            title="⚠ DELETE ACCOUNT",
            content=layout,
            size_hint=(0.9, 0.6),
            auto_dismiss=False
        )

        def do_delete(_):
            password = ti_password.text
            if not password:
                return

            try:
                username = getattr(self.manager, 'admin_name', '')
                with sqlite3.connect(DB_NAME) as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT id FROM users WHERE username=? AND password=?",
                        (username, password)
                    )
                    if not c.fetchone():
                        Popup(
                            title="Error",
                            content=Label(text="Incorrect password"),
                            size_hint=(0.6, 0.3)
                        ).open()
                        return

                    c.execute("DELETE FROM users WHERE username=?", (username,))
                    conn.commit()

                popup.dismiss()
                log_action("ACCOUNT_DELETE", f"Account {username} deleted", username)
                
                # Logout
                self.manager.current = "login"
                
                Popup(
                    title="✓ Account Deleted",
                    content=Label(text="Your account has been deleted.\n\nYou can now register a new admin."),
                    size_hint=(0.7, 0.4)
                ).open()
            except Exception as e:
                print(f"Delete account error: {e}")

        btn_delete.bind(on_release=do_delete)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def toggle_theme(self, theme):
        """Toggle dark/light theme"""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("UPDATE settings SET value=? WHERE key='theme'", (theme.lower(),))
                conn.commit()
            
            Popup(
                title="Theme Changed",
                content=Label(text=f"Theme set to {theme}\n\nRestart app to apply."),
                size_hint=(0.7, 0.3)
            ).open()
        except Exception as e:
            print(f"Toggle theme error: {e}")


# ========== LOADING SCREEN ==========

class LoadingScreen(Screen):
    message = StringProperty("Loading...")

    def start_loading(self, next_screen, message="Loading..."):
        self.message = message
        self.ids.loading_bar.value = 0
        Clock.schedule_once(lambda dt: self.simulate_loading(next_screen), 0.1)

    def simulate_loading(self, next_screen):
        def update_progress(dt):
            self.ids.loading_bar.value += 20
            if self.ids.loading_bar.value >= 100:
                Clock.unschedule(update_progress)
                Clock.schedule_once(lambda dt: setattr(self.manager, 'current', next_screen), 0.2)
        
        Clock.schedule_interval(update_progress, 0.1)


# ========== CONTACTS SCREEN (PLACEHOLDER) ==========

class ContactsScreen(Screen):
    """This is actually the Customers screen - redirect"""
    def on_enter(self):
        # Redirect to customers
        self.manager.current = "customers"


# ========== MAIN APP CLASS ==========

class BookKeepApp(App):
    sidebar_icon = StringProperty("icons/home.png")
    admin_name = StringProperty("")
    admin_role = StringProperty("admin")
    admin_id = NumericProperty(0)

    def build(self):
        self.title = "NCJM Stockflow - Enterprise Edition"
        
        # Initialize database
        if not init_database():
            print("⚠ Database initialization failed")
        
        # Schedule auto-backup (daily at 2 AM)
        Clock.schedule_interval(lambda dt: self.check_auto_backup(), 3600)  # Check every hour
        
        return Builder.load_file("main.kv")

    def on_start(self):
        """Called when app starts"""
        print("=" * 50)
        print("🚀 NCJM Stockflow - Enterprise Edition")
        print("=" * 50)
        print(f"✅ Database: {DB_NAME}")
        print(f"✅ Export Dir: {EXPORT_DIR}")
        print(f"✅ Backup Dir: {BACKUP_DIR}")
        print(f"✅ Receipt Dir: {RECEIPT_DIR}")
        print(f"📷 Barcode Scanner: {'Enabled' if BARCODE_AVAILABLE else 'Disabled'}")
        print(f"📄 PDF Receipts: {'Enabled' if PDF_AVAILABLE else 'Disabled'}")
        print(f"📊 Charts: {'Enabled' if CHARTS_AVAILABLE else 'Disabled'}")
        print(f"🤖 Forecasting: {'Enabled' if FORECASTING_AVAILABLE else 'Disabled'}")
        print("=" * 50)

    def check_auto_backup(self):
        """Check if it's time for auto-backup (2 AM)"""
        current_hour = datetime.now().hour
        if current_hour == 2:
            auto_backup()

    def switch_screen(self, screen_name):
        """Switch screen and update sidebar icon"""
        try:
            self.root.current = screen_name
            icons = {
                "home": "icons/home.png",
                "dashboard": "icons/dashboard.png",
                "transaction": "icons/transaction.png",
                "inventory": "icons/inventory.png",
                "contacts": "icons/contacts.png",
                "admin": "icons/admin.png",
            }
            self.sidebar_icon = icons.get(screen_name, "icons/home.png")
        except Exception as e:
            print(f"Switch screen error: {e}")

    def logout_user(self):
        """Logout current user"""
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        layout.add_widget(Label(text="Are you sure you want to logout?", halign="center"))

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        btn_yes = Button(text="Yes, Logout")
        btn_no = Button(text="Cancel")
        btn_row.add_widget(btn_yes)
        btn_row.add_widget(btn_no)
        layout.add_widget(btn_row)

        popup = Popup(title="Logout", content=layout, size_hint=(0.7, 0.4))

        def do_logout(_):
            log_action("LOGOUT", f"{self.admin_name} logged out", self.admin_name)
            self.admin_name = ""
            self.admin_role = ""
            self.admin_id = 0
            popup.dismiss()
            self.root.current = "login"

        btn_yes.bind(on_release=do_logout)
        btn_no.bind(on_release=lambda *_: popup.dismiss())
        popup.open()


# ========== SCREEN MANAGER ==========

class ScreenManagement(ScreenManager):
    admin_name = StringProperty("")
    admin_role = StringProperty("admin")
    admin_id = NumericProperty(0)


# ========== RUN APP ==========

if __name__ == "__main__":
    try:
        BookKeepApp().run()
    except Exception as e:
        print(f"❌ App crashed: {e}")
        import traceback
        traceback.print_exc()
 