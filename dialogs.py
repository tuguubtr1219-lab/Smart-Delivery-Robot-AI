import tkinter as tk
from tkinter import messagebox


class SaveMapDialog(tk.Toplevel):
    """Modal form for saving map information."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Save Map Information")
        self.configure(bg="#111827")
        self.resizable(False, False)
        self.result = None
        self.transient(parent)
        self.grab_set()

        wrapper = tk.Frame(self, bg="#111827", padx=22, pady=20)
        wrapper.pack(fill="both", expand=True)

        tk.Label(wrapper, text="Save Mapping Session", bg="#111827", fg="#F9FAFB", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        self.name_var = tk.StringVar(value="Tuan")
        self.note_var = tk.StringVar(value="Indoor mapping test")
        self.autosave_var = tk.StringVar(value="0")

        self._add_field(wrapper, 1, "User / Operator name", self.name_var)
        self._add_field(wrapper, 2, "Project note", self.note_var)
        self._add_field(wrapper, 3, "Autosave minutes (0 = off)", self.autosave_var)

        btns = tk.Frame(wrapper, bg="#111827")
        btns.grid(row=4, column=0, columnspan=2, sticky="e", pady=(18, 0))
        tk.Button(btns, text="Cancel", command=self._cancel, bg="#374151", fg="#F9FAFB", activebackground="#4B5563", relief="flat", padx=18, pady=8, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        tk.Button(btns, text="Save", command=self._ok, bg="#2563EB", fg="#FFFFFF", activebackground="#1D4ED8", relief="flat", padx=22, pady=8, font=("Segoe UI", 10, "bold")).pack(side="left")

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())
        self.wait_window(self)

    def _add_field(self, parent, row, label, var):
        tk.Label(parent, text=label, bg="#111827", fg="#CBD5E1", font=("Segoe UI", 10)).grid(row=row, column=0, sticky="w", pady=7)
        entry = tk.Entry(parent, textvariable=var, width=42, bg="#1F2937", fg="#F9FAFB", insertbackground="#F9FAFB", relief="flat", font=("Segoe UI", 10))
        entry.grid(row=row, column=1, sticky="ew", padx=(14, 0), pady=7, ipady=7)

    def _ok(self):
        name = self.name_var.get().strip()
        note = self.note_var.get().strip()
        try:
            autosave = max(0.0, float(self.autosave_var.get().strip() or "0"))
        except ValueError:
            messagebox.showerror("Invalid value", "Autosave minutes must be a number.", parent=self)
            return
        if not name:
            messagebox.showerror("Missing name", "Please enter the operator name.", parent=self)
            return
        self.result = {"user_name": name, "project_note": note, "autosave_minutes": autosave}
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.grab_release()
        self.destroy()


class AddPersonDialog(tk.Toplevel):
    """Modal form for registering a face profile."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Register Person")
        self.configure(bg="#111827")
        self.resizable(False, False)
        self.result = None
        self.transient(parent)
        self.grab_set()

        wrapper = tk.Frame(self, bg="#111827", padx=22, pady=20)
        wrapper.pack(fill="both", expand=True)

        tk.Label(wrapper, text="Register Face Profile", bg="#111827", fg="#F9FAFB", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        self.name_var = tk.StringVar()
        tk.Label(wrapper, text="Person name", bg="#111827", fg="#CBD5E1", font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w")
        tk.Entry(wrapper, textvariable=self.name_var, width=36, bg="#1F2937", fg="#F9FAFB", insertbackground="#F9FAFB", relief="flat", font=("Segoe UI", 10)).grid(row=1, column=1, padx=(14, 0), ipady=7)

        btns = tk.Frame(wrapper, bg="#111827")
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(18, 0))
        tk.Button(btns, text="Cancel", command=self._cancel, bg="#374151", fg="#F9FAFB", relief="flat", padx=18, pady=8).pack(side="left", padx=(0, 10))
        tk.Button(btns, text="Register", command=self._ok, bg="#7C3AED", fg="#FFFFFF", relief="flat", padx=22, pady=8).pack(side="left")

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())
        self.wait_window(self)

    def _ok(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Missing name", "Please enter a name.", parent=self)
            return
        self.result = name
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.grab_release()
        self.destroy()
