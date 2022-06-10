import pickle
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Dataset(object):
    def __init__(self) -> None:
        self.__filename = ""
        self.__filepath = ""
        self.__DataFrame = None
        self.__DTClassifier = None
        self.__RFClassifier = None
        self.__KNNClassifier = None
        self.__SVMClassifier = None
        self.__BoostClassifier = None
        self.__LogisticRegressionClassifier = None

        self.__DTreeClassifier_filename = "DTree_RandomSearch.sav"
        self.__RFClassifier_filename = "RForest_RandomSearch.sav"
        self.__KNNClassifier_filename = "KNN_GridSearch.sav"
        self.__SVMClassifier_filename = "SVM_RandomSearch.sav"
        self.__BoostClassifier_filename = "GradBoost_GridSearch.sav"
        self.__LogisticRegressionClassifier_filename = "LogReg_GridSearch.sav"

    def read_file(self, filepath: str):
        """Чтение файла с данными"""
        self.__filename = filepath.split("/")[-1]
        self.__filepath = filepath
        df = pd.read_excel(filepath)
        if "Target" in df.columns:
            df = df.drop("Target", axis=1)
        self.__DataFrame = df

    def get_feature_names(self):
        """Получить имена свойств"""
        return list(self.__DataFrame.columns[:-1])

    def get_features(self):
        return self.__DataFrame.loc[:, "v1":"v25"]

    def get_class_names(self):
        """Получить имена классов"""
        return list(
            map(
                lambda x: "Class " + str(x),
                np.unique(self.__DataFrame.iloc[:, -1:]),
            )
        )

    def load_DTClassifier(self):
        """Загрузка дерева решений"""
        self.__DTClassifier = pickle.load(open(self.__DTreeClassifier_filename, "rb"))

    def load_RFClassifier(self):
        """Загрузка случайного леса"""
        self.__RFClassifier = pickle.load(open(self.__RFClassifier_filename, "rb"))

    def load_KNNClassifier(self, n: int = 5):
        """Загрузка KNN-классификатора"""
        self.__KNNClassifier = pickle.load(open(self.__KNNClassifier_filename, "rb"))

    def load_SVMClassifier(self):
        """Загрузка SVM классификатора"""
        self.__SVMClassifier = pickle.load(open(self.__SVMClassifier_filename, "rb"))

    def load_BoostClassifier(self):
        """Загрузка бустинга"""
        self.__BoostClassifier = pickle.load(
            open(self.__BoostClassifier_filename, "rb")
        )

    def load_LogisticRegressionClassifier(self):
        """Загрузка логистической регрессии"""
        self.__LogisticRegressionClassifier = pickle.load(
            open(self.__LogisticRegressionClassifier_filename, "rb")
        )

    def predict_with_DTClassifier(self):
        df = self.get_features()
        df = df.replace("", None).replace("nan", None).fillna(0)
        X = df.values

        Y_pred = self.DTreeClassifier.predict(X)

        return Y_pred

    def predict_with_RFClassifier(self):
        df = self.get_features()
        df = df.replace("", None).replace("nan", None).fillna(0)
        X = df

        Y_pred = self.RFClassifier.predict(X)

        return Y_pred

    def predict_with_KNNClassifier(self):
        df = self.get_features()
        df = df.replace("", None).replace("nan", None).fillna(0)
        X = df.values

        Y_pred = self.KNNClassifier.predict(X)

        return Y_pred

    def predict_with_SVMClassifier(self):
        df = self.get_features()
        df = df.replace("", None).replace("nan", None).fillna(0)
        X = df.values

        Y_pred = self.SVMClassifier.predict(X)

        return Y_pred

    def predict_with_BoostClassifier(self):
        df = self.get_features()
        df = df.replace("", None).replace("nan", None).fillna(0)
        X = df

        Y_pred = self.BoostClassifier.predict(X)

        return Y_pred

    def predict_with_LogisticRegressionClassifier(self):
        df = self.get_features()
        df = df.replace("", None).replace("nan", None).fillna(0)
        X = df

        Y_pred = self.LogisticRegressionClassifier.predict(X)

        return Y_pred

    @property
    def filename(self):
        return self.__filename

    @property
    def filepath(self):
        return self.__filepath

    @property
    def DataFrame(self):
        return self.__DataFrame

    @property
    def DTreeClassifier(self):
        return self.__DTClassifier

    @property
    def RFClassifier(self):
        return self.__RFClassifier

    @property
    def KNNClassifier(self):
        return self.__KNNClassifier

    @property
    def SVMClassifier(self):
        return self.__SVMClassifier

    @property
    def BoostClassifier(self):
        return self.__BoostClassifier

    @property
    def LogisticRegressionClassifier(self):
        return self.__LogisticRegressionClassifier

    @property
    def DTree_filename(self):
        return self.__DTreeClassifier_filename

    @property
    def RF_filename(self):
        return self.__RFClassifier_filename

    @property
    def KNN_filename(self):
        return self.__KNNClassifier_filename

    @property
    def SVM_filename(self):
        return self.__SVMClassifier_filename

    @property
    def BoostClassifier_filename(self):
        return self.__BoostClassifier_filename

    @property
    def LogisticRegressionClassifier_filename(self):
        return self.__LogisticRegressionClassifier_filename

    @DataFrame.setter
    def DataFrame(self, DataFrame: pd.DataFrame):
        self.__DataFrame = DataFrame

    @DTreeClassifier.setter
    def DTreeClassifier(self, DTreeClassifier: DecisionTreeClassifier):
        self.__DTClassifier = DTreeClassifier

    @RFClassifier.setter
    def RFClassifier(self, RFClassifier: RandomForestClassifier):
        self.__RFClassifier = RFClassifier

    @KNNClassifier.setter
    def KNNClassifier(self, KNNClassifier: KNeighborsClassifier):
        self.__KNNClassifier = KNNClassifier

    @SVMClassifier.setter
    def SVMClassifier(self, SVMClassifier: SVC):
        self.__SVMClassifier = SVMClassifier

    @BoostClassifier.setter
    def BoostClassifier(self, BoostClassifier: GradientBoostingClassifier):
        self.__BoostClassifier = BoostClassifier

    @LogisticRegressionClassifier.setter
    def LogisticRegressionClassifier(self, LogRegClassifer: LogisticRegression):
        self.__LogisticRegressionClassifier = LogRegClassifer


class Tableview(ttk.Treeview):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind("<Double-1>", lambda event: self.onDoubleClick(event))
        self.bind("<Delete>", lambda event: self.onDelete(event))
        self.scroll_v = ttk.Scrollbar(self, command=self.yview)
        self.scroll_v.pack(side="right", fill="y")
        self.config(yscrollcommand=self.scroll_v.set)
        self.scroll_h = ttk.Scrollbar(self, orient="horizontal", command=self.xview)
        self.scroll_h.pack(side="bottom", fill="x")
        self.config(xscrollcommand=self.scroll_h.set)

    def onDoubleClick(self, event):
        """Executed, when a row is double-clicked. Opens
        read-only EntryPopup above the item's column, so it is possible
        to select text"""

        # close previous popups
        try:  # in case there was no previous popup
            self.entryPopup.destroy()
        except AttributeError:
            pass

        # what row and column was clicked on
        rowid = self.identify_row(event.y)
        column = self.identify_column(event.x)
        column = self.column(column, "id")

        # handle exception when header is double click
        if not rowid:
            return

        # get column position info
        x, y, width, height = self.bbox(rowid, column)

        # y-axis offset
        pady = height // 2

        # place Entry popup properly
        text = self.item(rowid, "values")[int(column[1:]) - 1]
        self.entryPopup = EntryPopup(self, rowid, int(column[1:]) - 1, text)
        self.entryPopup.place(x=x, y=y + pady, width=width, height=height, anchor="w")

    def onDelete(self, event):
        selected_items = self.selection()
        for item in selected_items:
            self.delete(item)

    def fill_table_headers(self, params_num: int):
        headings = [f"v{num+1}" for num in range(params_num)]
        self.config(columns=headings)
        for _, param in enumerate(headings):
            self.heading(param, text=param)
            self.column(param, width=70, minwidth=70)

    def set_table_content(self, data: pd.DataFrame):
        for rowId in self.get_children():
            self.delete(rowId)

        X = data.iloc[:, :].values

        for row in X:
            self.insert("", "end", values=list(row))

    def get_table_content(self):
        table_content = []
        table_rows = self.get_children()
        for row in table_rows:
            table_content.append([])
            row_values = self.item(row)["values"]
            for value in row_values:
                table_content[-1].append(value)

        df = pd.DataFrame(table_content, columns=self["columns"], dtype="object")
        return df

    def set_columns_visibility(self, columns: list):
        self.configure(displaycolumns=columns)
        for column in columns:
            self.column(column, width=50)


class EntryPopup(ttk.Entry):
    def __init__(self, parent, iid, column, text, **kw):
        ttk.Style().configure("pad.TEntry", padding="1 1 1 1")
        super().__init__(parent, style="pad.TEntry", **kw)
        self.tv = parent
        self.iid = iid
        self.column = column

        self.insert(0, text)
        self["exportselection"] = False

        self.focus_force()
        self.select_all()
        self.bind("<Return>", self.on_return)
        self.bind("<Control-a>", self.select_all)
        self.bind("<Escape>", lambda *ignore: self.destroy())

    def on_return(self, event):
        try:
            rowid = self.tv.focus()
            item = [*self.tv.item(rowid, "values")]
            item[self.column] = self.get()
            self.tv.insert("", self.tv.index(rowid), values=item)
            self.tv.delete(rowid)
            self.destroy()
        except Exception as e:
            print("ERROR")
            print(e)

    def select_all(self, *ignore):
        """Set selection on the whole text"""
        self.selection_range(0, "end")

        # returns 'break' to interrupt default key-bindings
        return "break"


class MainActivity(tk.Tk):
    def __init__(self, *args, **kwargs):
        self.Dataset = Dataset()

        self.__ui__()

        self.DtreeBtn.invoke()

    def __ui__(self):
        super().__init__()

        self.title("Сухоруков К.Е. ИНМО-03-21")
        self.__params_amount = 25
        self.frame_color = "#f7889d"
        self.button_color = "#88f7e2"
        self.filetypes = {"Excel .xlsx"}
        self.Dataset.load_DTClassifier()
        self.Dataset.load_SVMClassifier()

        # UI options
        self.paddings = {"padx": 10, "pady": 5}
        self.entry_font = {"font": ("Cambria", 15)}
        self.minsize(1000, 800)

        self.__ui_menu__()

        self.BtnFrame = tk.Frame(relief="raised", bg=self.frame_color)

        self.__ui_add_buttons__(self.BtnFrame)

        self.BtnFrame.pack(fill="x")

        self.TableFrame = tk.Frame(relief="raised", bg=self.frame_color)

        self.__ui_add_table(self.TableFrame)
        self.TableFrame.pack(side="top", pady=15, expand=True, fill="both")

    def __ui_add_buttons__(self, master=None):

        self.chosen_model = tk.StringVar(master, value="Дерево решений")

        self.DtreeBtn = tk.Radiobutton(
            master,
            text="Дерево решений",
            command=self.transform_table,
            font=self.entry_font,
            value="Дерево решений",
            variable=self.chosen_model,
            background=self.button_color,
        )
        self.DtreeBtn.pack(self.paddings, anchor="nw", side="left")

        # self.RFBtn = tk.Radiobutton(
        #     master,
        #     text="Случайный лес",
        #     command=self.transform_table,
        #     font=self.entry_font,
        #     value="Случайный лес",
        #     variable=self.chosen_model,
        #     background=self.button_color,
        # )
        # self.RFBtn.pack(self.paddings, anchor="nw", side="left")

        # self.KNNBtn = tk.Radiobutton(
        #     master,
        #     text="Метод К-ближайших соседей",
        #     command=self.transform_table,
        #     font=self.entry_font,
        #     value="Метод К-ближайших соседей",
        #     variable=self.chosen_model,
        #     background=self.button_color
        # )
        # self.KNNBtn.pack(self.paddings, anchor="nw", side="left")

        self.SVMBtn = tk.Radiobutton(
            master,
            text="Метод опорных векторов",
            command=self.transform_table,
            font=self.entry_font,
            value="Метод опорных векторов",
            variable=self.chosen_model,
            background=self.button_color,
        )
        self.SVMBtn.pack(self.paddings, anchor="nw", side="left")

        # self.BoostBtn = tk.Radiobutton(
        #     master,
        #     text="Бустинг",
        #     command=self.transform_table,
        #     font=self.entry_font,
        #     value="Бустинг",
        #     variable=self.chosen_model,
        #     background=self.button_color,
        # )
        # self.BoostBtn.pack(self.paddings, anchor="nw", side="left")

        # self.LogRegBtn = tk.Radiobutton(
        #     master,
        #     text="Логистическая регрессия",
        #     command=self.transform_table,
        #     font=self.entry_font,
        #     value="Логистическая регрессия",
        #     variable=self.chosen_model,
        #     background=self.button_color,
        # )
        # self.LogRegBtn.pack(self.paddings, anchor="nw", side="left")

    def __ui_add_table(self, master=None):
        self.Table = Tableview(master, show="headings")
        self.Table.heading("#0", text="№")
        self.Table.column("#0", width=0)

        self.Table.fill_table_headers(self.__params_amount)
        self.Table.pack(side="top", expand=True, fill="both")

        self.addRowButton = tk.Button(
            self.TableFrame,
            text="Добавить строку",
            command=self.addRowBtn_event,
            font=self.entry_font,
            background=self.button_color,
        )
        self.addRowButton.pack(side="right", anchor="center")

        self.predictButton = tk.Button(
            self.TableFrame,
            text="Предсказать по данным таблицы",
            command=self.predictBtn_event,
            font=self.entry_font,
            background=self.button_color,
        )
        self.predictButton.pack(side="left", anchor="center")

    def __ui_menu__(self):
        """Создание и настройка строки меню"""

        self.menu = tk.Menu(self)
        self.menu.add_command(
            label="Открыть файл с рабочими данными",
            command=self.open_file_event,
            font=self.entry_font,
        )
        self.configure(menu=self.menu)

    def __ui_widget_children_state(self, widget, new_state: str):
        try:
            children = widget.winfo_children()
            for child in children:
                child.config(state=new_state)
        except Exception as e:
            print("Error during setting state of widget's children")
            print(e)

    def open_file_event(self):
        filename = fd.askopenfilename(filetypes=self.filetypes)
        try:
            self.Dataset.read_file(filename)
            self.__ui_widget_children_state(self.BtnFrame, "active")
            self.Table.set_table_content(self.Dataset.DataFrame)
        except FileNotFoundError as e:
            print("File not found")
            print(e)

    def predictBtn_event(self):
        self.Dataset.DataFrame = self.Table.get_table_content()

        predict_handlers = {
            "Дерево решений": self.classifier1_predict,
            "Метод опорных векторов": self.classifier2_predict,
        }

        chosen_handler = predict_handlers[self.chosen_model.get()]
        chosen_handler()

    def classifier1_predict(self):
        try:
            y_pred = self.Dataset.predict_with_DTClassifier()
            y_pred = pd.DataFrame(y_pred, columns=["Target"])
            original_file = self.Dataset.filename
            stacked_df = pd.concat([self.Dataset.DataFrame, y_pred], axis=1)
            self.output_to_excel(f"{original_file}_DTree_prediction.xlsx", stacked_df)
        except Exception as e:
            print(e)

    def classifier2_predict(self):
        try:
            y_pred = self.Dataset.predict_with_SVMClassifier()
            y_pred = pd.DataFrame(y_pred, columns=["Target"])
            stacked_df = pd.concat([self.Dataset.DataFrame, y_pred], axis=1)
            original_file = self.Dataset.filename
            self.output_to_excel(
                f"{original_file}_SVM_prediction.xlsx", stacked_df
            )
        except Exception as e:
            print(e)

    def output_to_excel(self, filename: str, data: pd.DataFrame):
        try:
            df = pd.read_excel(filename)
            new_data = df.append(data)
            new_data.to_excel(filename, index=False, encoding="cp1251")
        except FileNotFoundError as e:
            data.to_excel(filename, index=False, encoding="cp1251")

    def addRowBtn_event(self):
        self.Table.insert("", "end", values=[""] * self.__params_amount)

    def transform_table(self):

        shown_columns = {
            "Дерево решений": [
                "v2",
                "v3",
                "v5",
                "v7",
                "v8",
                "v9",
                "v10",
                "v12",
                "v15",
                "v16",
                "v17",
                "v19",
                "v20",
                "v22",
                "v23",
                "v24",
            ],
            "Метод опорных векторов": [
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "v6",
                "v7",
                "v8",
                "v9",
                "v10",
                "v11",
                "v12",
                "v13",
                "v14",
                "v15",
                "v16",
                "v17",
                "v18",
                "v19",
                "v20",
                "v21",
                "v22",
                "v23",
                "v24",
                "v25",
            ],
        }
        chosen_model = self.chosen_model.get()
        self.Table.set_columns_visibility(shown_columns[chosen_model])


if __name__ == "__main__":
    print("БАЗА 9 -- Предсказание рецидива заболевания")
    app = MainActivity()
    app.mainloop()
