import json
import csv
import io
import asyncio
import logging
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple

import google.generativeai as genai
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    ReplyKeyboardMarkup, 
    KeyboardButton, 
    InlineKeyboardMarkup, 
    InlineKeyboardButton, 
    ReplyKeyboardRemove
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

# ======================== НАСТРОЙКИ ========================
BOT_TOKEN = "your-telegram-token"
DATA_DIR = "data"
BACKUP_DIR = "backups"

# ======================== НАСТРОЙКИ GEMINI AI ========================
GEMINI_API_KEY = "your-telegram-key"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================== КОНСТАНТЫ ========================
class Roles:
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class GradeTypes:
    EXAM = "Экзамен"
    CREDIT = "Зачет"
    COURSEWORK = "Курсовая"
    LAB = "Лабораторная"
    HOMEWORK = "Домашняя работа"
    TEST = "Тест"
    QUIZ = "Контрольная"
    
    @classmethod
    def all(cls):
        return [cls.EXAM, cls.CREDIT, cls.COURSEWORK, cls.LAB, 
                cls.HOMEWORK, cls.TEST, cls.QUIZ]

class AttendanceStatus:
    PRESENT = "Присутствовал"
    ABSENT = "Отсутствовал"
    LATE = "Опоздал"
    EXCUSED = "Уважительная причина"

# ======================== КЛАСС ДЛЯ РАБОТЫ С ДАННЫМИ ========================
class DataHandler:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.backup_dir = BACKUP_DIR
        self.users_file = os.path.join(self.data_dir, "users.csv")
        self.groups_file = os.path.join(self.data_dir, "groups.csv")
        self.subjects_file = os.path.join(self.data_dir, "subjects.csv")
        self.grades_file = os.path.join(self.data_dir, "grades.csv")
        self.attendance_file = os.path.join(self.data_dir, "attendance.csv")
        self.teacher_subjects_file = os.path.join(self.data_dir, "teacher_subjects.csv")
        
        self._init_files()
        self._create_backup()
    
    def _init_files(self):
        """Создает папки и файлы если их нет"""
        for directory in [self.data_dir, self.backup_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Создана папка: {directory}")
        
        files = {
            self.users_file: ['user_id', 'telegram_id', 'full_name', 'role', 'group_id', 'created_at', 'is_active'],
            self.groups_file: ['group_id', 'group_name', 'created_at'],
            self.subjects_file: ['subject_id', 'subject_name', 'created_at'],
            self.grades_file: ['id', 'student_id', 'subject_id', 'teacher_id', 'grade', 'grade_type', 'date', 'comment'],
            self.attendance_file: ['id', 'student_id', 'subject_id', 'teacher_id', 'date', 'status', 'comment'],
            self.teacher_subjects_file: ['id', 'teacher_id', 'subject_id', 'group_id']
        }
        
        for file_path, headers in files.items():
            if not os.path.exists(file_path):
                df = pd.DataFrame(columns=headers)
                df.to_csv(file_path, index=False, encoding='utf-8')
                logger.info(f"Создан файл: {file_path}")
    
    def _create_backup(self):
        """Создает резервную копию данных"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = os.path.join(self.backup_dir, timestamp)
            
            if not os.path.exists(backup_subdir):
                os.makedirs(backup_subdir)
            
            files = [
                self.users_file, self.groups_file, self.subjects_file,
                self.grades_file, self.attendance_file, self.teacher_subjects_file
            ]
            
            for file_path in files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    backup_path = os.path.join(backup_subdir, filename)
                    df = self._read_csv(file_path)
                    df.to_csv(backup_path, index=False, encoding='utf-8')
            
            logger.info(f"Создана резервная копия: {backup_subdir}")
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
    
    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """Чтение CSV файла с обработкой ошибок"""
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return pd.read_csv(file_path, encoding='utf-8')
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка чтения {file_path}: {e}")
            return pd.DataFrame()
    
    def _write_csv(self, file_path: str, df: pd.DataFrame) -> bool:
        """Запись в CSV файл с обработкой ошибок"""
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            logger.error(f"Ошибка записи {file_path}: {e}")
            return False
    
    def _append_to_csv(self, file_path: str, data_dict: dict) -> bool:
        """Добавление записи в CSV"""
        try:
            df = self._read_csv(file_path)
            new_df = pd.DataFrame([data_dict])
            df = pd.concat([df, new_df], ignore_index=True)
            return self._write_csv(file_path, df)
        except Exception as e:
            logger.error(f"Ошибка добавления в {file_path}: {e}")
            return False
    
    def get_user(self, telegram_id: int) -> Optional[Dict]:
        """Получение пользователя по telegram_id"""
        df = self._read_csv(self.users_file)
        if df.empty:
            return None
        
        user = df[df['telegram_id'] == telegram_id]
        return user.iloc[0].to_dict() if not user.empty else None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Получение пользователя по user_id (не telegram_id)"""
        df = self._read_csv(self.users_file)
        if df.empty:
            return None
        
        user = df[df['user_id'] == user_id]
        return user.iloc[0].to_dict() if not user.empty else None
    
    def update_user(self, telegram_id: int, **kwargs) -> bool:
        """Обновление данных пользователя"""
        try:
            df = self._read_csv(self.users_file)
            if df.empty:
                return False
            
            mask = df['telegram_id'] == telegram_id
            for key, value in kwargs.items():
                if key in df.columns:
                    df.loc[mask, key] = value
            
            return self._write_csv(self.users_file, df)
        except Exception as e:
            logger.error(f"Ошибка обновления пользователя: {e}")
            return False
    
    def register_user(self, telegram_id: int, full_name: str, role: str, group_id: Optional[int] = None) -> Optional[int]:
        """Регистрация нового пользователя"""
        try:
            df = self._read_csv(self.users_file)
            user_id = int(df['user_id'].max() + 1) if not df.empty and not df['user_id'].isna().all() else 1
            
            user_data = {
                'user_id': user_id,
                'telegram_id': int(telegram_id),
                'full_name': str(full_name),
                'role': str(role),
                'group_id': group_id if group_id else np.nan,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'is_active': True
            }
            
            if self._append_to_csv(self.users_file, user_data):
                logger.info(f"Зарегистрирован пользователь: {full_name} (ID: {user_id})")
                return user_id
            return None
        except Exception as e:
            logger.error(f"Ошибка регистрации пользователя: {e}")
            return None
    
    def get_groups(self) -> List[Dict]:
        """Получение всех групп"""
        df = self._read_csv(self.groups_file)
        return df.to_dict('records') if not df.empty else []
    
    def get_group_by_name(self, group_name: str) -> Optional[Dict]:
        """Получение группы по имени"""
        df = self._read_csv(self.groups_file)
        if df.empty:
            return None
        
        group = df[df['group_name'] == group_name]
        return group.iloc[0].to_dict() if not group.empty else None
    
    def create_group(self, group_name: str) -> Optional[int]:
        """Создание новой группы"""
        try:
            # Проверяем, существует ли уже группа
            existing = self.get_group_by_name(group_name)
            if existing:
                logger.warning(f"Группа {group_name} уже существует")
                return existing['group_id']
            
            df = self._read_csv(self.groups_file)
            group_id = int(df['group_id'].max() + 1) if not df.empty and not df['group_id'].isna().all() else 1
            
            group_data = {
                'group_id': group_id,
                'group_name': str(group_name),
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if self._append_to_csv(self.groups_file, group_data):
                logger.info(f"Создана группа: {group_name} (ID: {group_id})")
                return group_id
            return None
        except Exception as e:
            logger.error(f"Ошибка создания группы: {e}")
            return None
    
    def get_students_in_group(self, group_id: int) -> List[Dict]:
        """Получение студентов в группе"""
        df = self._read_csv(self.users_file)
        if df.empty:
            return []
        
        students = df[(df['group_id'] == group_id) & (df['role'] == Roles.STUDENT) & (df['is_active'] == True)]
        return students.to_dict('records')
    
    def get_subjects(self) -> List[Dict]:
        """Получение всех предметов"""
        df = self._read_csv(self.subjects_file)
        return df.to_dict('records') if not df.empty else []
    
    def add_grade(self, student_id: int, subject_id: int, teacher_id: int, 
                  grade: float, grade_type: str, comment: str = "") -> Optional[int]:
        """Добавление оценки"""
        try:
            df = self._read_csv(self.grades_file)
            grade_id = int(df['id'].max() + 1) if not df.empty and not df['id'].isna().all() else 1
            
            grade_data = {
                'id': grade_id,
                'student_id': int(student_id),
                'subject_id': int(subject_id),
                'teacher_id': int(teacher_id),
                'grade': float(grade),
                'grade_type': str(grade_type),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'comment': str(comment)
            }
            
            if self._append_to_csv(self.grades_file, grade_data):
                logger.info(f"Добавлена оценка: студент {student_id}, оценка {grade}")
                return grade_id
            return None
        except Exception as e:
            logger.error(f"Ошибка добавления оценки: {e}")
            return None
    
    def get_student_grades(self, student_id: int, subject_id: Optional[int] = None) -> List[Dict]:
        """Получение оценок студента"""
        df = self._read_csv(self.grades_file)
        if df.empty:
            return []
        
        if subject_id:
            grades = df[(df['student_id'] == student_id) & (df['subject_id'] == subject_id)]
        else:
            grades = df[df['student_id'] == student_id]
        
        return grades.to_dict('records')
    
    def get_group_statistics(self, group_id: int) -> Dict:
        """Получение статистики по группе"""
        students = self.get_students_in_group(group_id)
        
        if not students:
            return {
                'total_students': 0,
                'avg_grade': 0,
                'total_grades': 0
            }
        
        all_grades = []
        for student in students:
            grades = self.get_student_grades(student['user_id'])
            all_grades.extend([g['grade'] for g in grades])
        
        return {
            'total_students': len(students),
            'avg_grade': round(sum(all_grades) / len(all_grades), 2) if all_grades else 0,
            'total_grades': len(all_grades)
        }
    
    def create_test_data(self):
        """Создание тестовых данных"""
        try:
            # Создаем группы
            groups_df = self._read_csv(self.groups_file)
            if groups_df.empty:
                test_groups = [
                    'ИТ-201', 'ИТ-202', 'ПИ-301', 'ПИ-302', 'ИБ-401'
                ]
                for group_name in test_groups:
                    self.create_group(group_name)
            
            # Создаем предметы
            subjects_df = self._read_csv(self.subjects_file)
            if subjects_df.empty:
                subjects = [
                    {'subject_id': 1, 'subject_name': 'Программирование', 'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    {'subject_id': 2, 'subject_name': 'Математика', 'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    {'subject_id': 3, 'subject_name': 'Базы данных', 'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    {'subject_id': 4, 'subject_name': 'Английский язык', 'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    {'subject_id': 5, 'subject_name': 'Физика', 'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    {'subject_id': 6, 'subject_name': 'Алгоритмы', 'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    {'subject_id': 7, 'subject_name': 'Веб-разработка', 'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                ]
                for subject in subjects:
                    self._append_to_csv(self.subjects_file, subject)
            
            logger.info("✅ Тестовые данные созданы")
        except Exception as e:
            logger.error(f"Ошибка создания тестовых данных: {e}")

# ======================== ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ ========================
db = DataHandler()
db.create_test_data()

# ======================== AI АНАЛИТИК С GEMINI ========================
class AnalyticsAgent:
    def __init__(self):
        self.model = gemini_model
        self.db = db
    
    def analyze_student(self, student_id: int) -> Dict:
        """Анализ успеваемости студента с AI-подобными рекомендациями"""
        grades = self.db.get_student_grades(student_id)
        
        if not grades:
            return {
                'risk': 'unknown',
                'message': '📭 Недостаточно данных для анализа. Начните получать оценки!',
                'recommendations': []
            }
        
        # Подготавливаем данные для AI
        total_grades = len(grades)
        avg_grade = sum(g['grade'] for g in grades) / total_grades
        recent_grades = [g['grade'] for g in sorted(grades, key=lambda x: x['date'])[-5:]]
        recent_avg = sum(recent_grades) / len(recent_grades) if recent_grades else avg_grade
        
        # Анализ по типам работ
        grade_types = {}
        for g in grades:
            gtype = g['grade_type']
            if gtype not in grade_types:
                grade_types[gtype] = []
            grade_types[gtype].append(g['grade'])
        
        weak_areas = []
        strong_areas = []
        
        for gtype, glist in grade_types.items():
            avg = sum(glist) / len(glist)
            if avg < 3.5:
                weak_areas.append(f"{gtype} (средний балл: {avg:.2f})")
            elif avg >= 4.5:
                strong_areas.append(f"{gtype} (средний балл: {avg:.2f})")
        
        # Создаем промпт для Gemini
        prompt = f"""
Ты - опытный образовательный аналитик. Проанализируй успеваемость студента и дай персонализированные рекомендации.

ДАННЫЕ СТУДЕНТА:
- Всего оценок: {total_grades}
- Средний балл: {avg_grade:.2f}
- Средний балл за последние 5 оценок: {recent_avg:.2f}
- Последние оценки: {', '.join(map(str, recent_grades))}
- Слабые стороны: {', '.join(weak_areas) if weak_areas else 'нет'}
- Сильные стороны: {', '.join(strong_areas) if strong_areas else 'нет'}

Детали по типам работ:
{self._format_grade_types(grade_types)}

ЗАДАНИЕ:
1. Оцени общий уровень успеваемости (отлично/хорошо/удовлетворительно/требует внимания/критический)
2. Определи тренд (улучшение/ухудшение/стабильность)
3. Дай 3-5 конкретных, практических рекомендаций для улучшения
4. Укажи риски (если есть)

Ответь КРАТКО и ПО ДЕЛУ. Используй эмодзи для наглядности.
"""
        
        try:
            # Запрос к Gemini
            response = self.model.generate_content(prompt)
            ai_analysis = response.text
            
            # Определяем уровень риска на основе среднего балла
            if avg_grade < 3.0:
                risk = 'high'
                emoji = '🔴'
                message = 'Критический уровень успеваемости'
            elif avg_grade < 3.5:
                risk = 'medium'
                emoji = '🟡'
                message = 'Успеваемость требует внимания'
            elif avg_grade < 4.5:
                risk = 'low'
                emoji = '🟢'
                message = 'Хорошая успеваемость'
            else:
                risk = 'excellent'
                emoji = '⭐'
                message = 'Отличная успеваемость!'
            
            return {
                'risk': risk,
                'emoji': emoji,
                'message': message,
                'avg_grade': round(avg_grade, 2),
                'recent_avg': round(recent_avg, 2),
                'total_grades': total_grades,
                'ai_analysis': ai_analysis,
                'weak_areas': weak_areas,
                'strong_areas': strong_areas
            }
            
        except Exception as e:
            logger.error(f"Ошибка AI анализа: {e}")
            # Fallback на базовый анализ
            return self._basic_analysis(avg_grade, recent_avg, total_grades, weak_areas, strong_areas)
    
    def _format_grade_types(self, grade_types: Dict) -> str:
        """Форматирование типов оценок для промпта"""
        result = []
        for gtype, glist in grade_types.items():
            avg = sum(glist) / len(glist)
            result.append(f"- {gtype}: {len(glist)} оценок, средний балл {avg:.2f}")
        return "\n".join(result)
    
    def _basic_analysis(self, avg_grade, recent_avg, total_grades, weak_areas, strong_areas) -> Dict:
        """Базовый анализ без AI (fallback)"""
        if avg_grade < 3.0:
            risk = 'high'
            emoji = '🔴'
            message = 'Критический уровень успеваемости'
            recommendations = [
                "🎯 Обратитесь к преподавателю за консультацией",
                "📚 Увеличьте время на подготовку",
                "👥 Попросите помощи у одногруппников"
            ]
        elif avg_grade < 3.5:
            risk = 'medium'
            emoji = '🟡'
            message = 'Успеваемость требует внимания'
            recommendations = [
                "📊 Проанализируйте свои ошибки",
                "⏰ Планируйте время на учебу",
                "💪 Уделите внимание слабым местам"
            ]
        elif avg_grade < 4.5:
            risk = 'low'
            emoji = '🟢'
            message = 'Хорошая успеваемость'
            recommendations = [
                "📈 Продолжайте в том же духе!",
                "🎯 Стремитесь к отличным оценкам",
                "🌟 Развивайте сильные стороны"
            ]
        else:
            risk = 'excellent'
            emoji = '⭐'
            message = 'Отличная успеваемость!'
            recommendations = [
                "🏆 Отличная работа!",
                "🤝 Помогайте другим студентам",
                "🎓 Участвуйте в олимпиадах"
            ]
        
        return {
            'risk': risk,
            'emoji': emoji,
            'message': message,
            'avg_grade': round(avg_grade, 2),
            'recent_avg': round(recent_avg, 2),
            'total_grades': total_grades,
            'ai_analysis': "\n".join(recommendations),
            'weak_areas': weak_areas,
            'strong_areas': strong_areas
        }
    
    def get_help_response(self, student_id: int, question: str) -> str:
        """Получение помощи от AI"""
        # Получаем данные студента
        user = self.db.get_user_by_id(student_id)
        grades = self.db.get_student_grades(student_id)
        
        if not user:
            return "❌ Не удалось найти информацию о студенте."
        
        # Подготавливаем контекст
        context = f"""
ИНФОРМАЦИЯ О СТУДЕНТЕ:
- ФИО: {user.get('full_name', 'Неизвестно')}
- Группа: {user.get('group_id', 'Не указана')}
- Всего оценок: {len(grades) if grades else 0}
"""
        
        if grades:
            avg_grade = sum(g['grade'] for g in grades) / len(grades)
            context += f"- Средний балл: {avg_grade:.2f}\n"
        
        # Создаем промпт
        prompt = f"""
Ты - помощник студента в образовательной системе. Отвечай дружелюбно, профессионально и по делу.

{context}

ВОПРОС СТУДЕНТА:
{question}

Дай краткий, полезный ответ. Используй эмодзи для наглядности.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Ошибка получения помощи от AI: {e}")
            return "❌ Извините, не удалось получить ответ. Попробуйте переформулировать вопрос или обратитесь к преподавателю."
    
    def get_statistics_analysis(self, student_id: int) -> str:
        """Детальный анализ статистики студента"""
        grades = self.db.get_student_grades(student_id)
        
        if not grades:
            return "📭 У вас пока нет оценок для анализа статистики."
        
        # Анализ по предметам
        subjects = self.db.get_subjects()
        subject_dict = {s['subject_id']: s['subject_name'] for s in subjects}
        
        grades_by_subject = {}
        for grade in grades:
            subj_id = grade['subject_id']
            if subj_id not in grades_by_subject:
                grades_by_subject[subj_id] = []
            grades_by_subject[subj_id].append(grade['grade'])
        
        # Анализ по типам работ
        grades_by_type = {}
        for grade in grades:
            gtype = grade['grade_type']
            if gtype not in grades_by_type:
                grades_by_type[gtype] = []
            grades_by_type[gtype].append(grade['grade'])
        
        # Временной анализ
        recent_30_days = []
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        for grade in grades:
            if grade['date'] >= thirty_days_ago:
                recent_30_days.append(grade['grade'])
        
        # Создаем промпт для детального анализа
        prompt = f"""
Проанализируй детальную статистику успеваемости студента и дай развернутый отчет.

ОБЩАЯ СТАТИСТИКА:
- Всего оценок: {len(grades)}
- Средний балл: {sum(g['grade'] for g in grades) / len(grades):.2f}
- Оценок за последние 30 дней: {len(recent_30_days)}
{f"- Средний балл за 30 дней: {sum(recent_30_days) / len(recent_30_days):.2f}" if recent_30_days else ""}

СТАТИСТИКА ПО ПРЕДМЕТАМ:
{self._format_subject_stats(grades_by_subject, subject_dict)}

СТАТИСТИКА ПО ТИПАМ РАБОТ:
{self._format_type_stats(grades_by_type)}

ЗАДАНИЕ:
1. Проанализируй сильные и слабые предметы
2. Оцени прогресс за последний месяц
3. Дай рекомендации по каждому слабому предмету
4. Предложи стратегию улучшения общей успеваемости

Ответ должен быть структурированным и практичным. Используй эмодзи.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Ошибка AI анализа статистики: {e}")
            return self._basic_statistics(grades, grades_by_subject, subject_dict, grades_by_type)
    
    def _format_subject_stats(self, grades_by_subject: Dict, subject_dict: Dict) -> str:
        """Форматирование статистики по предметам"""
        result = []
        for subj_id, grade_list in grades_by_subject.items():
            subj_name = subject_dict.get(subj_id, f"Предмет {subj_id}")
            avg = sum(grade_list) / len(grade_list)
            result.append(f"- {subj_name}: {len(grade_list)} оценок, средний {avg:.2f}")
        return "\n".join(result)
    
    def _format_type_stats(self, grades_by_type: Dict) -> str:
        """Форматирование статистики по типам работ"""
        result = []
        for gtype, grade_list in grades_by_type.items():
            avg = sum(grade_list) / len(grade_list)
            result.append(f"- {gtype}: {len(grade_list)} оценок, средний {avg:.2f}")
        return "\n".join(result)
    
    def _basic_statistics(self, grades, grades_by_subject, subject_dict, grades_by_type) -> str:
        """Базовая статистика без AI"""
        result = "📊 *ДЕТАЛЬНАЯ СТАТИСТИКА*\n\n"
        
        result += "📚 *По предметам:*\n"
        for subj_id, grade_list in sorted(grades_by_subject.items(), 
                                         key=lambda x: sum(x[1])/len(x[1]), 
                                         reverse=True):
            subj_name = subject_dict.get(subj_id, f"Предмет {subj_id}")
            avg = sum(grade_list) / len(grade_list)
            emoji = "⭐" if avg >= 4.5 else "✅" if avg >= 4.0 else "⚠️" if avg >= 3.5 else "❌"
            result += f"{emoji} {subj_name}: {avg:.2f} ({len(grade_list)} оценок)\n"
        
        result += "\n📋 *По типам работ:*\n"
        for gtype, grade_list in sorted(grades_by_type.items(), 
                                       key=lambda x: sum(x[1])/len(x[1]), 
                                       reverse=True):
            avg = sum(grade_list) / len(grade_list)
            emoji = "⭐" if avg >= 4.5 else "✅" if avg >= 4.0 else "⚠️" if avg >= 3.5 else "❌"
            result += f"{emoji} {gtype}: {avg:.2f} ({len(grade_list)} оценок)\n"
        
        return result

# ======================== ИНИЦИАЛИЗАЦИЯ БОТА ========================
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
analytics = AnalyticsAgent()

# ======================== СОСТОЯНИЯ ========================
class RegistrationForm(StatesGroup):
    role = State()
    full_name = State()
    group = State()
    creating_group = State()

class GradeForm(StatesGroup):
    select_group = State()
    select_subject = State()
    select_student = State()
    select_type = State()
    enter_grade = State()
    enter_comment = State()

class ProfileForm(StatesGroup):
    edit_name = State()
    change_group = State()

class HelpForm(StatesGroup):
    waiting_for_question = State()

class AttendanceForm(StatesGroup):
    select_group = State()
    select_subject = State()
    select_date = State()
    mark_attendance = State()
    process_attendance = State()

# ======================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========================

def get_cancel_keyboard():
    """Универсальная клавиатура отмены"""
    builder = ReplyKeyboardBuilder()
    builder.button(text="❌ Отмена")
    return builder.as_markup(resize_keyboard=True)

def get_role_keyboard():
    """Клавиатура выбора роли"""
    builder = ReplyKeyboardBuilder()
    builder.button(text="👨‍🎓 Студент")
    builder.button(text="👨‍🏫 Преподаватель")
    builder.button(text="❌ Отмена")
    builder.adjust(2, 1)
    return builder.as_markup(resize_keyboard=True, one_time_keyboard=True)

def get_groups_keyboard(include_create: bool = True):
    """Клавиатура выбора группы"""
    groups = db.get_groups()
    builder = ReplyKeyboardBuilder()
    
    for group in groups[:8]:
        builder.button(text=group['group_name'])
    
    if include_create:
        builder.button(text="➕ Создать группу")
    builder.button(text="❌ Отмена")
    
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True, one_time_keyboard=True)

def get_subjects_keyboard():
    """Клавиатура выбора предмета"""
    subjects = db.get_subjects()
    builder = ReplyKeyboardBuilder()
    
    for subject in subjects[:10]:
        builder.button(text=subject['subject_name'])
    
    builder.button(text="❌ Отмена")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True, one_time_keyboard=True)

def get_student_menu():
    """Меню для студента"""
    builder = ReplyKeyboardBuilder()
    builder.button(text="📊 Мои оценки")
    builder.button(text="🎯 Анализ успеваемости")
    builder.button(text="👤 Мой профиль")
    builder.button(text="📈 Статистика")
    builder.button(text="ℹ️ Помощь")
    builder.button(text="🚪 Выход")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)

def get_teacher_menu():
    """Меню для преподавателя"""
    builder = ReplyKeyboardBuilder()
    builder.button(text="➕ Поставить оценку")
    builder.button(text="👥 Список студентов")
    builder.button(text="📈 Аналитика группы")
    builder.button(text="👤 Мой профиль")
    builder.button(text="ℹ️ Помощь")
    builder.button(text="🚪 Выход")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)

def get_grade_type_keyboard():
    """Типы оценок"""
    builder = ReplyKeyboardBuilder()
    for grade_type in GradeTypes.all():
        builder.button(text=grade_type)
    builder.button(text="❌ Отмена")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True, one_time_keyboard=True)

def get_profile_keyboard():
    """Меню профиля"""
    builder = ReplyKeyboardBuilder()
    builder.button(text="✏️ Изменить имя")
    builder.button(text="🔄 Сменить группу")
    builder.button(text="🔙 Назад")
    builder.adjust(2, 1)
    return builder.as_markup(resize_keyboard=True)

def get_confirm_keyboard():
    """Клавиатура подтверждения"""
    builder = ReplyKeyboardBuilder()
    builder.button(text="✅ Да")
    builder.button(text="❌ Нет")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True, one_time_keyboard=True)

# ======================== ОСНОВНЫЕ ОБРАБОТЧИКИ ========================

@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    """Начало работы с ботом"""
    await state.clear()
    user = db.get_user(message.from_user.id)
    
    if user and user.get('is_active'):
        role_emoji = "👨‍🎓" if user['role'] == Roles.STUDENT else "👨‍🏫"
        role_name = "Студент" if user['role'] == Roles.STUDENT else "Преподаватель"
        
        menu = get_student_menu() if user['role'] == Roles.STUDENT else get_teacher_menu()
        
        await message.answer(
            f"{role_emoji} Добро пожаловать, {user['full_name']}!\n\n"
            f"🎭 Роль: {role_name}\n"
            f"📅 Зарегистрирован: {user['created_at'][:10]}\n\n"
            f"Выберите действие из меню:",
            reply_markup=menu
        )
    else:
        await message.answer(
            "👋 Добро пожаловать в Learning Analytics Agent!\n\n"
            "🎓 Я помогу вам:\n"
            "• Отслеживать успеваемость\n"
            "• Анализировать прогресс\n"
            "• Получать персональные рекомендации\n\n"
            "Давайте начнем регистрацию!\n"
            "Выберите вашу роль:",
            reply_markup=get_role_keyboard()
        )
        await state.set_state(RegistrationForm.role)

# ======================== РЕГИСТРАЦИЯ ========================

@dp.message(RegistrationForm.role)
async def process_role(message: types.Message, state: FSMContext):
    """Обработка выбора роли"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer(
            "Регистрация отменена. Для начала работы используйте /start",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    if message.text not in ["👨‍🎓 Студент", "👨‍🏫 Преподаватель"]:
        await message.answer(
            "❌ Пожалуйста, выберите роль из предложенных вариантов.",
            reply_markup=get_role_keyboard()
        )
        return
    
    role = Roles.STUDENT if message.text == "👨‍🎓 Студент" else Roles.TEACHER
    await state.update_data(role=role)
    
    await message.answer(
        "📝 Введите ваше ФИО полностью\n"
        "(например: Иванов Иван Иванович):",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(RegistrationForm.full_name)

@dp.message(RegistrationForm.full_name)
async def process_full_name(message: types.Message, state: FSMContext):
    """Обработка ФИО"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer(
            "Регистрация отменена. Для начала работы используйте /start",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    full_name = message.text.strip()
    
    if len(full_name) < 5 or not any(char.isalpha() for char in full_name):
        await message.answer(
            "❌ Пожалуйста, введите корректное ФИО (минимум 5 символов).",
            reply_markup=get_cancel_keyboard()
        )
        return
    
    await state.update_data(full_name=full_name)
    data = await state.get_data()
    
    if data['role'] == Roles.STUDENT:
        await message.answer(
            "🎓 Выберите вашу учебную группу\n"
            "или создайте новую:",
            reply_markup=get_groups_keyboard()
        )
        await state.set_state(RegistrationForm.group)
    else:
        # Регистрируем преподавателя
        user_id = db.register_user(
            telegram_id=message.from_user.id,
            full_name=full_name,
            role=data['role']
        )
        
        if user_id:
            await message.answer(
                f"✅ Регистрация успешно завершена!\n\n"
                f"👤 ФИО: {full_name}\n"
                f"🎭 Роль: Преподаватель\n"
                f"🆔 ID: {user_id}\n\n"
                f"Теперь вы можете выставлять оценки и просматривать аналитику.",
                reply_markup=get_teacher_menu()
            )
            logger.info(f"Зарегистрирован преподаватель: {full_name}")
        else:
            await message.answer(
                "❌ Ошибка регистрации. Попробуйте снова /start",
                reply_markup=ReplyKeyboardRemove()
            )
        
        await state.clear()

@dp.message(RegistrationForm.group)
async def process_group(message: types.Message, state: FSMContext):
    """Обработка выбора группы"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer(
            "Регистрация отменена. Для начала работы используйте /start",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    if message.text == "➕ Создать группу":
        await message.answer(
            "📝 Введите название новой группы\n"
            "(например: ИТ-201):",
            reply_markup=get_cancel_keyboard()
        )
        await state.set_state(RegistrationForm.creating_group)
        return
    
    group_name = message.text.strip()
    group = db.get_group_by_name(group_name)
    
    if not group:
        await message.answer(
            "❌ Группа не найдена. Выберите из списка или создайте новую.",
            reply_markup=get_groups_keyboard()
        )
        return
    
    data = await state.get_data()
    user_id = db.register_user(
        telegram_id=message.from_user.id,
        full_name=data['full_name'],
        role=data['role'],
        group_id=group['group_id']
    )
    
    if user_id:
        await message.answer(
            f"✅ Регистрация успешно завершена!\n\n"
            f"👤 ФИО: {data['full_name']}\n"
            f"🎭 Роль: Студент\n"
            f"👥 Группа: {group_name}\n"
            f"🆔 ID: {user_id}\n\n"
            f"Добро пожаловать в систему!",
            reply_markup=get_student_menu()
        )
        logger.info(f"Зарегистрирован студент: {data['full_name']} ({group_name})")
    else:
        await message.answer(
            "❌ Ошибка регистрации. Попробуйте снова /start",
            reply_markup=ReplyKeyboardRemove()
        )
    
    await state.clear()

@dp.message(RegistrationForm.creating_group)
async def process_creating_group(message: types.Message, state: FSMContext):
    """Создание новой группы"""
    if message.text == "❌ Отмена":
        await message.answer(
            "Выберите существующую группу:",
            reply_markup=get_groups_keyboard()
        )
        await state.set_state(RegistrationForm.group)
        return
    
    group_name = message.text.strip().upper()
    
    if len(group_name) < 2 or len(group_name) > 20:
        await message.answer(
            "❌ Название группы должно быть от 2 до 20 символов.",
            reply_markup=get_cancel_keyboard()
        )
        return
    
    group_id = db.create_group(group_name)
    
    if group_id:
        data = await state.get_data()
        user_id = db.register_user(
            telegram_id=message.from_user.id,
            full_name=data['full_name'],
            role=data['role'],
            group_id=group_id
        )
        
        if user_id:
            await message.answer(
                f"✅ Группа создана и регистрация завершена!\n\n"
                f"👤 ФИО: {data['full_name']}\n"
                f"🎭 Роль: Студент\n"
                f"👥 Группа: {group_name} (новая)\n"
                f"🆔 ID: {user_id}",
                reply_markup=get_student_menu()
            )
            logger.info(f"Создана группа {group_name} и зарегистрирован студент")
        else:
            await message.answer("❌ Ошибка регистрации.")
    else:
        await message.answer("❌ Ошибка создания группы.")
    
    await state.clear()

# ======================== ФУНКЦИИ СТУДЕНТА ========================

@dp.message(F.text == "📊 Мои оценки")
async def show_grades(message: types.Message):
    """Показать оценки студента"""
    user = db.get_user(message.from_user.id)
    
    if not user or user['role'] != Roles.STUDENT:
        await message.answer("❌ Эта функция доступна только студентам.")
        return
    
    grades = db.get_student_grades(user['user_id'])
    
    if not grades:
        await message.answer(
            "📭 У вас пока нет оценок.\n\n"
            "Оценки будут появляться здесь после того,\n"
            "как преподаватели начнут их выставлять."
        )
        return
    
    # Группируем по предметам
    subjects = db.get_subjects()
    subject_dict = {s['subject_id']: s['subject_name'] for s in subjects}
    
    grades_by_subject = {}
    for grade in grades:
        subj_id = grade['subject_id']
        if subj_id not in grades_by_subject:
            grades_by_subject[subj_id] = []
        grades_by_subject[subj_id].append(grade)
    
    response = "📊 *Ваши оценки по предметам:*\n\n"
    
    for subj_id, subj_grades in grades_by_subject.items():
        subj_name = subject_dict.get(subj_id, f"Предмет {subj_id}")
        avg = sum(g['grade'] for g in subj_grades) / len(subj_grades)
        
        response += f"📚 *{subj_name}*\n"
        response += f"   Средний балл: {avg:.2f}\n"
        response += f"   Оценок: {len(subj_grades)}\n"
        
        # Последние 3 оценки
        recent = sorted(subj_grades, key=lambda x: x['date'], reverse=True)[:3]
        for g in recent:
            response += f"   • {g['grade_type']}: {g['grade']} ({g['date']})\n"
        response += "\n"
    
    # Общая статистика
    all_grades_values = [g['grade'] for g in grades]
    overall_avg = sum(all_grades_values) / len(all_grades_values)
    
    response += f"📈 *Общая статистика:*\n"
    response += f"Средний балл: {overall_avg:.2f}\n"
    response += f"Всего оценок: {len(grades)}"
    
    await message.answer(response, parse_mode="Markdown")

@dp.message(F.text == "🎯 Анализ успеваемости")
async def show_analysis(message: types.Message):
    """Показать AI анализ успеваемости"""
    user = db.get_user(message.from_user.id)
    
    if not user or user['role'] != Roles.STUDENT:
        await message.answer("❌ Эта функция доступна только студентам.")
        return
    
    await message.answer("🔄 Анализирую вашу успеваемость с помощью AI...")
    
    analysis = analytics.analyze_student(user['user_id'])
    
    response = f"{analysis['emoji']} *{analysis['message']}*\n\n"
    
    if 'avg_grade' in analysis:
        response += f"📊 *Показатели:*\n"
        response += f"• Средний балл: {analysis['avg_grade']}\n"
        response += f"• Последние оценки: {analysis['recent_avg']}\n"
        response += f"• Всего оценок: {analysis['total_grades']}\n\n"
    
    if analysis.get('weak_areas'):
        response += "⚠️ *Слабые стороны:*\n"
        for area in analysis['weak_areas'][:3]:
            response += f"• {area}\n"
        response += "\n"
    
    if analysis.get('strong_areas'):
        response += "💪 *Сильные стороны:*\n"
        for area in analysis['strong_areas'][:3]:
            response += f"• {area}\n"
        response += "\n"
    
    if analysis.get('ai_analysis'):
        response += f"🤖 *AI Анализ и рекомендации:*\n{analysis['ai_analysis']}\n"
    
    await message.answer(response, parse_mode="Markdown")

@dp.message(F.text == "📈 Статистика")
async def show_statistics(message: types.Message):
    """Показать детальную статистику с AI анализом"""
    user = db.get_user(message.from_user.id)
    
    if not user or user['role'] != Roles.STUDENT:
        await message.answer("❌ Эта функция доступна только студентам.")
        return
    
    await message.answer("🔄 Анализирую вашу статистику с помощью AI...")
    
    ai_statistics = analytics.get_statistics_analysis(user['user_id'])
    
    response = "📈 *СТАТИСТИКА С AI АНАЛИЗОМ*\n\n"
    response += ai_statistics
    
    # Разбиваем на части если слишком длинное
    if len(response) > 4000:
        parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for part in parts:
            await message.answer(part, parse_mode="Markdown")
    else:
        await message.answer(response, parse_mode="Markdown")

# ======================== ВЫСТАВЛЕНИЕ ОЦЕНОК ========================

@dp.message(F.text == "➕ Поставить оценку")
async def start_grade_process(message: types.Message, state: FSMContext):
    """Начать процесс выставления оценки"""
    user = db.get_user(message.from_user.id)
    
    if not user or user['role'] != Roles.TEACHER:
        await message.answer("❌ Эта функция доступна только преподавателям.")
        return
    
    await state.update_data(teacher_id=user['user_id'])
    await message.answer(
        "1️⃣ Выберите группу студента:",
        reply_markup=get_groups_keyboard(include_create=False)
    )
    await state.set_state(GradeForm.select_group)

@dp.message(GradeForm.select_group)
async def process_select_group(message: types.Message, state: FSMContext):
    """Выбор группы для оценки"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer("Отменено.", reply_markup=get_teacher_menu())
        return
    
    group = db.get_group_by_name(message.text.strip())
    
    if not group:
        await message.answer(
            "❌ Группа не найдена.",
            reply_markup=get_groups_keyboard(include_create=False)
        )
        return
    
    students = db.get_students_in_group(group['group_id'])
    
    if not students:
        await message.answer(
            f"❌ В группе {message.text} нет студентов.\n"
            "Выберите другую группу:",
            reply_markup=get_groups_keyboard(include_create=False)
        )
        return
    
    await state.update_data(group_id=group['group_id'], group_name=message.text)
    
    await message.answer(
        "2️⃣ Выберите предмет:",
        reply_markup=get_subjects_keyboard()
    )
    await state.set_state(GradeForm.select_subject)

@dp.message(GradeForm.select_subject)
async def process_select_subject(message: types.Message, state: FSMContext):
    """Выбор предмета"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer("Отменено.", reply_markup=get_teacher_menu())
        return
    
    subjects = db.get_subjects()
    subject = None
    
    for s in subjects:
        if s['subject_name'] == message.text.strip():
            subject = s
            break
    
    if not subject:
        await message.answer(
            "❌ Предмет не найден.",
            reply_markup=get_subjects_keyboard()
        )
        return
    
    await state.update_data(subject_id=subject['subject_id'], subject_name=message.text)
    
    data = await state.get_data()
    students = db.get_students_in_group(data['group_id'])
    
    builder = ReplyKeyboardBuilder()
    for student in students[:15]:
        builder.button(text=student['full_name'])
    builder.button(text="❌ Отмена")
    builder.adjust(2)
    
    await message.answer(
        "3️⃣ Выберите студента:",
        reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True)
    )
    await state.set_state(GradeForm.select_student)

@dp.message(GradeForm.select_student)
async def process_select_student(message: types.Message, state: FSMContext):
    """Выбор студента"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer("Отменено.", reply_markup=get_teacher_menu())
        return
    
    data = await state.get_data()
    students = db.get_students_in_group(data['group_id'])
    student = None
    
    for s in students:
        if s['full_name'] == message.text.strip():
            student = s
            break
    
    if not student:
        await message.answer("❌ Студент не найден.")
        return
    
    await state.update_data(student_id=student['user_id'], student_name=message.text)
    
    await message.answer(
        "4️⃣ Выберите тип оценки:",
        reply_markup=get_grade_type_keyboard()
    )
    await state.set_state(GradeForm.select_type)

@dp.message(GradeForm.select_type)
async def process_select_type(message: types.Message, state: FSMContext):
    """Выбор типа оценки"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer("Отменено.", reply_markup=get_teacher_menu())
        return
    
    if message.text not in GradeTypes.all():
        await message.answer(
            "❌ Неверный тип оценки.",
            reply_markup=get_grade_type_keyboard()
        )
        return
    
    await state.update_data(grade_type=message.text)
    
    await message.answer(
        "5️⃣ Введите оценку (от 1 до 5):\n"
        "Можно использовать дробные числа (например: 4.5)",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(GradeForm.enter_grade)

@dp.message(GradeForm.enter_grade)
async def process_enter_grade(message: types.Message, state: FSMContext):
    """Ввод оценки"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer("Отменено.", reply_markup=get_teacher_menu())
        return
    
    try:
        grade = float(message.text.strip().replace(',', '.'))
        
        if not (1 <= grade <= 5):
            raise ValueError("Оценка вне диапазона")
        
        await state.update_data(grade=grade)
        
        await message.answer(
            "6️⃣ Добавить комментарий? (необязательно)\n"
            "Введите комментарий или нажмите 'Пропустить':",
            reply_markup=ReplyKeyboardBuilder()
            .button(text="⏭️ Пропустить")
            .button(text="❌ Отмена")
            .adjust(2)
            .as_markup(resize_keyboard=True, one_time_keyboard=True)
        )
        await state.set_state(GradeForm.enter_comment)
        
    except ValueError:
        await message.answer(
            "❌ Пожалуйста, введите число от 1 до 5\n"
            "(можно использовать дробные: 4.5, 3.7 и т.д.)",
            reply_markup=get_cancel_keyboard()
        )

@dp.message(GradeForm.enter_comment)
async def process_enter_comment(message: types.Message, state: FSMContext):
    """Ввод комментария"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer("Отменено.", reply_markup=get_teacher_menu())
        return
    
    comment = "" if message.text == "⏭️ Пропустить" else message.text.strip()
    
    data = await state.get_data()
    
    grade_id = db.add_grade(
        student_id=data['student_id'],
        subject_id=data['subject_id'],
        teacher_id=data['teacher_id'],
        grade=data['grade'],
        grade_type=data['grade_type'],
        comment=comment
    )
    
    if grade_id:
        response = (
            "✅ *Оценка успешно добавлена!*\n\n"
            f"👤 Студент: {data['student_name']}\n"
            f"👥 Группа: {data['group_name']}\n"
            f"📚 Предмет: {data['subject_name']}\n"
            f"📋 Тип: {data['grade_type']}\n"
            f"📊 Оценка: {data['grade']}\n"
            f"📅 Дата: {datetime.now().strftime('%d.%m.%Y')}"
        )
        
        if comment:
            response += f"\n💬 Комментарий: {comment}"
        
        await message.answer(response, parse_mode="Markdown", reply_markup=get_teacher_menu())
        logger.info(f"Оценка {data['grade']} добавлена студенту {data['student_name']}")
    else:
        await message.answer(
            "❌ Ошибка при добавлении оценки. Попробуйте снова.",
            reply_markup=get_teacher_menu()
        )
    
    await state.clear()

# ======================== ФУНКЦИИ ПРЕПОДАВАТЕЛЯ ========================

@dp.message(F.text == "👥 Список студентов")
async def show_students_list(message: types.Message):
    """Показать список студентов"""
    user = db.get_user(message.from_user.id)
    
    if not user or user['role'] != Roles.TEACHER:
        await message.answer("❌ Эта функция доступна только преподавателям.")
        return
    
    groups = db.get_groups()
    
    if not groups:
        await message.answer("📭 В системе пока нет групп.")
        return
    
    response = "👥 *Список студентов по группам:*\n\n"
    
    for group in groups[:10]:
        students = db.get_students_in_group(group['group_id'])
        response += f"📚 *{group['group_name']}* ({len(students)} студентов)\n"
        
        if students:
            for student in students[:5]:
                grades = db.get_student_grades(student['user_id'])
                avg = sum(g['grade'] for g in grades) / len(grades) if grades else 0
                response += f"   • {student['full_name']}"
                if grades:
                    response += f" (ср. балл: {avg:.2f})"
                response += "\n"
            
            if len(students) > 5:
                response += f"   ... и еще {len(students) - 5}\n"
        else:
            response += "   Нет студентов\n"
        
        response += "\n"
    
    await message.answer(response, parse_mode="Markdown")

@dp.message(F.text == "📈 Аналитика группы")
async def show_group_analytics(message: types.Message):
    """Показать аналитику группы"""
    user = db.get_user(message.from_user.id)
    
    if not user or user['role'] != Roles.TEACHER:
        await message.answer("❌ Эта функция доступна только преподавателям.")
        return
    
    groups = db.get_groups()
    
    if not groups:
        await message.answer("📭 В системе нет групп.")
        return
    
    response = "📈 *Аналитика по группам:*\n\n"
    
    for group in groups[:10]:
        stats = db.get_group_statistics(group['group_id'])
        
        response += f"📚 *{group['group_name']}*\n"
        response += f"   Студентов: {stats['total_students']}\n"
        
        if stats['total_students'] > 0:
            response += f"   Средний балл: {stats['avg_grade']}\n"
            response += f"   Всего оценок: {stats['total_grades']}\n"
            
            emoji = "🟢" if stats['avg_grade'] >= 4 else "🟡" if stats['avg_grade'] >= 3.5 else "🔴"
            response += f"   Статус: {emoji}\n"
        
        response += "\n"
    
    await message.answer(response, parse_mode="Markdown")

# ======================== ПРОФИЛЬ ========================

@dp.message(F.text == "👤 Мой профиль")
async def show_profile(message: types.Message):
    """Показать профиль пользователя"""
    user = db.get_user(message.from_user.id)
    
    if not user:
        await message.answer("❌ Профиль не найден. Используйте /start")
        return
    
    role_emoji = "👨‍🎓" if user['role'] == Roles.STUDENT else "👨‍🏫"
    role_name = "Студент" if user['role'] == Roles.STUDENT else "Преподаватель"
    
    response = f"{role_emoji} *Ваш профиль*\n\n"
    response += f"👤 ФИО: {user['full_name']}\n"
    response += f"🎭 Роль: {role_name}\n"
    response += f"🆔 ID: {user['user_id']}\n"
    response += f"📅 Регистрация: {user['created_at'][:10]}\n"
    
    if user['role'] == Roles.STUDENT and user.get('group_id'):
        groups = db.get_groups()
        group = next((g for g in groups if g['group_id'] == user['group_id']), None)
        if group:
            response += f"👥 Группа: {group['group_name']}\n"
            
            grades = db.get_student_grades(user['user_id'])
            if grades:
                avg = sum(g['grade'] for g in grades) / len(grades)
                response += f"\n📊 Средний балл: {avg:.2f}\n"
                response += f"📈 Всего оценок: {len(grades)}"
    
    await message.answer(
        response,
        parse_mode="Markdown",
        reply_markup=get_profile_keyboard()
    )

@dp.message(F.text == "✏️ Изменить имя")
async def edit_name(message: types.Message, state: FSMContext):
    """Изменить имя"""
    await message.answer(
        "✏️ Введите новое ФИО:",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(ProfileForm.edit_name)

@dp.message(ProfileForm.edit_name)
async def process_edit_name(message: types.Message, state: FSMContext):
    """Обработка изменения имени"""
    if message.text == "❌ Отмена":
        await state.clear()
        user = db.get_user(message.from_user.id)
        menu = get_student_menu() if user['role'] == Roles.STUDENT else get_teacher_menu()
        await message.answer("Отменено.", reply_markup=menu)
        return
    
    new_name = message.text.strip()
    
    if len(new_name) < 5:
        await message.answer(
            "❌ ФИО должно содержать минимум 5 символов.",
            reply_markup=get_cancel_keyboard()
        )
        return
    
    if db.update_user(message.from_user.id, full_name=new_name):
        user = db.get_user(message.from_user.id)
        menu = get_student_menu() if user['role'] == Roles.STUDENT else get_teacher_menu()
        await message.answer(
            f"✅ ФИО успешно изменено на: {new_name}",
            reply_markup=menu
        )
        logger.info(f"Пользователь {message.from_user.id} изменил имя на {new_name}")
    else:
        await message.answer("❌ Ошибка обновления.")
    
    await state.clear()

@dp.message(F.text == "🔄 Сменить группу")
async def start_change_group(message: types.Message, state: FSMContext):
    """Начать смену группы"""
    user = db.get_user(message.from_user.id)
    
    if not user or user['role'] != Roles.STUDENT:
        await message.answer("❌ Эта функция доступна только студентам.")
        return
    
    await message.answer(
        "Выберите новую группу:",
        reply_markup=get_groups_keyboard(include_create=False)
    )
    await state.set_state(ProfileForm.change_group)

@dp.message(ProfileForm.change_group)
async def process_change_group(message: types.Message, state: FSMContext):
    """Обработка смены группы"""
    if message.text == "❌ Отмена":
        await state.clear()
        await message.answer("Отменено.", reply_markup=get_student_menu())
        return
    
    group = db.get_group_by_name(message.text.strip())
    
    if not group:
        await message.answer(
            "❌ Группа не найдена.",
            reply_markup=get_groups_keyboard(include_create=False)
        )
        return
    
    if db.update_user(message.from_user.id, group_id=group['group_id']):
        await message.answer(
            f"✅ Группа успешно изменена на: {message.text}",
            reply_markup=get_student_menu()
        )
        logger.info(f"Пользователь {message.from_user.id} сменил группу на {message.text}")
    else:
        await message.answer("❌ Ошибка обновления группы.")
    
    await state.clear()

@dp.message(F.text == "🔙 Назад")
async def go_back(message: types.Message):
    """Возврат в главное меню"""
    user = db.get_user(message.from_user.id)
    
    if not user:
        await message.answer("❌ Профиль не найден. Используйте /start")
        return
    
    menu = get_student_menu() if user['role'] == Roles.STUDENT else get_teacher_menu()
    await message.answer("Главное меню:", reply_markup=menu)

# ======================== ОБРАБОТЧИК ПОМОЩИ С AI ========================

@dp.message(F.text == "ℹ️ Помощь")
async def show_help(message: types.Message, state: FSMContext):
    """Показать помощь с возможностью задать вопрос AI"""
    user = db.get_user(message.from_user.id)
    
    if not user:
        await message.answer("❌ Профиль не найден.")
        return
    
    builder = ReplyKeyboardBuilder()
    builder.button(text="❓ Задать вопрос AI")
    builder.button(text="📚 Частые вопросы")
    builder.button(text="📖 Руководство")
    builder.button(text="🔙 Назад")
    builder.adjust(2, 2)
    
    help_text = """
ℹ️ *ПОМОЩЬ И ПОДДЕРЖКА*

*Возможности бота:*
- 📊 Просмотр оценок и успеваемости
- 🎯 AI-анализ вашего прогресса
- 📈 Детальная статистика
- 💡 Персональные рекомендации

*Нужна помощь?*
Задайте вопрос AI-помощнику - он ответит на любые вопросы об учебе, оценках и системе!

Выберите действие:
"""
    
    await message.answer(help_text, parse_mode="Markdown", 
                        reply_markup=builder.as_markup(resize_keyboard=True))

@dp.message(F.text == "❓ Задать вопрос AI")
async def ask_ai_question(message: types.Message, state: FSMContext):
    """Начать диалог с AI"""
    await message.answer(
        "🤖 *AI-Помощник готов ответить!*\n\n"
        "Задайте любой вопрос об учебе, оценках, успеваемости или работе системы.\n\n"
        "Например:\n"
        "• Как улучшить мои оценки по математике?\n"
        "• Почему мой средний балл снизился?\n"
        "• Что делать, если я отстаю по предмету?\n\n"
        "Напишите ваш вопрос:",
        parse_mode="Markdown",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(HelpForm.waiting_for_question)

@dp.message(HelpForm.waiting_for_question)
async def process_ai_question(message: types.Message, state: FSMContext):
    """Обработка вопроса к AI"""
    if message.text == "❌ Отмена":
        await state.clear()
        user = db.get_user(message.from_user.id)
        menu = get_student_menu() if user['role'] == Roles.STUDENT else get_teacher_menu()
        await message.answer("Отменено.", reply_markup=menu)
        return
    
    user = db.get_user(message.from_user.id)
    
    await message.answer("🤖 Думаю над ответом...")
    
    # Получаем ответ от AI
    ai_response = analytics.get_help_response(user['user_id'], message.text)
    
    response = f"🤖 *AI-Помощник отвечает:*\n\n{ai_response}\n\n"
    response += "❓ Есть еще вопросы? Просто напишите их или нажмите 'Назад'"
    
    await message.answer(response, parse_mode="Markdown")
    # Остаемся в том же состоянии для продолжения диалога

@dp.message(F.text == "📚 Частые вопросы")
async def show_faq(message: types.Message):
    """Показать частые вопросы"""
    faq = """
📚 *ЧАСТЫЕ ВОПРОСЫ*

*Q: Как посмотреть свои оценки?*
A: Нажмите кнопку "📊 Мои оценки" в главном меню

*Q: Как работает AI-анализ?*
A: AI анализирует ваши оценки, выявляет сильные и слабые стороны, дает персональные рекомендации

*Q: Можно ли изменить группу?*
A: Да, через "👤 Мой профиль" → "🔄 Сменить группу"

*Q: Как улучшить успеваемость?*
A: Используйте "🎯 Анализ успеваемости" для получения рекомендаций от AI

*Q: Кто видит мои оценки?*
A: Только вы и ваши преподаватели

Есть другие вопросы? Задайте их AI-помощнику!
"""
    
    await message.answer(faq, parse_mode="Markdown")

# ======================== ВЫХОД ИЗ СИСТЕМЫ ========================

@dp.message(F.text == "🚪 Выход")
async def cmd_logout(message: types.Message):
    """Выход из системы с возвратом к выбору роли"""
    user = db.get_user(message.from_user.id)
    
    if user:
        # Деактивируем пользователя
        db.update_user(message.from_user.id, is_active=False)
        
        await message.answer(
            "👋 Вы вышли из системы.\n\n"
            "Для входа выберите свою роль:",
            reply_markup=get_role_keyboard()
        )
        logger.info(f"Пользователь {user['full_name']} вышел из системы")
    else:
        await message.answer(
            "Вы не были авторизованы в системе.\n\n"
            "Выберите роль для входа:",
            reply_markup=get_role_keyboard()
        )

# ======================== ОБРАБОТКА НЕИЗВЕСТНЫХ КОМАНД ========================

@dp.message()
async def handle_unknown_commands(message: types.Message):
    """Обработка неизвестных команд"""
    user = db.get_user(message.from_user.id)
    
    if not user:
        await message.answer(
            "❌ Вы не зарегистрированы в системе.\n"
            "Используйте /start для регистрации."
        )
        return
    
    # Определяем текущее меню пользователя
    menu = get_student_menu() if user['role'] == Roles.STUDENT else get_teacher_menu()
    
    # Проверяем, есть ли текст в основном меню
    main_menu_options = []
    if user['role'] == Roles.STUDENT:
        main_menu_options = [
            "📊 Мои оценки", "🎯 Анализ успеваемости", "👤 Мой профиль",
            "📈 Статистика", "ℹ️ Помощь", "🚪 Выход"
        ]
    else:
        main_menu_options = [
            "➕ Поставить оценку", "👥 Список студентов", "📈 Аналитика группы",
            "👤 Мой профиль", "ℹ️ Помощь", "🚪 Выход"
        ]
    
    if message.text in main_menu_options:
        # Команда из главного меню уже обрабатывается другими хендлерами
        return
    
    await message.answer(
        "❓ *Неизвестная команда*\n\n"
        "Пожалуйста, выберите одну из доступных команд в меню ниже:",
        parse_mode="Markdown",
        reply_markup=menu
    )

# ======================== ЗАПУСК БОТА ========================

async def on_startup():
    """Действия при запуске бота"""
    logger.info("🤖 Бот запущен!")
    
    # Создаем директории если их нет
    for directory in [DATA_DIR, BACKUP_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Создана директория: {directory}")
    
    # Создаем тестовые данные если нужно
    if os.environ.get("CREATE_TEST_DATA", "false").lower() == "true":
        db.create_test_data()

async def on_shutdown():
    """Действия при остановке бота"""
    logger.info("🛑 Бот остановлен!")
    
    # Создаем резервную копию перед выходом
    db._create_backup()

async def main():
    """Основная функция запуска бота"""
    # Настраиваем диспетчер
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    
    # Запускаем бота
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

# ======================== ТОЧКА ВХОДА ========================

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Настраиваем логирование
    logger.info("=" * 50)
    logger.info("🚀 Learning Analytics Bot - Запуск...")
    logger.info(f"📁 Data directory: {DATA_DIR}")
    logger.info(f"💾 Backup directory: {BACKUP_DIR}")
    logger.info("=" * 50)
    
    try:
        # Запускаем бота
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise