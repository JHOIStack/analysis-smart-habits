from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
import uuid
from datetime import datetime

Base = declarative_base()


class Region(str, enum.Enum):
    NORTE = "NORTE"
    CENTRO = "CENTRO"
    SUR = "SUR"
    OCCIDENTE = "OCCIDENTE"
    SURESTE = "SURESTE"
    CDMX = "CDMX"
    INTERNACIONAL = "INTERNACIONAL"


class HabitCategory(str, enum.Enum):
    ENERGIA = "ENERGIA"
    AGUA = "AGUA"
    RESIDUOS = "RESIDUOS"
    MOVILIDAD = "MOVILIDAD"
    CONSUMO = "CONSUMO"


class HabitStatus(str, enum.Enum):
    ACTIVO = "ACTIVO"
    PAUSADO = "PAUSADO"
    COMPLETADO = "COMPLETADO"
    CANCELADO = "CANCELADO"


class ProfileType(str, enum.Enum):
    ECO_PRINCIPIANTE = "ECO_PRINCIPIANTE"
    ECO_AVANZADO = "ECO_AVANZADO"
    ECO_EXPERTO = "ECO_EXPERTO"
    
class InteractionType(str, enum.Enum):
    CLICK = "CLICK"
    IGNORE = "IGNORE"
    COMPLETE = "COMPLETE"
    SKIP = "SKIP"

class UserRole(str, enum.Enum):
    COMMON = "COMMON"
    SUPER = "SUPER"


class User(Base):
    __tablename__ = "User"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    age = Column(Integer)
    region = Column(Enum(Region))
    createdAt = Column(DateTime, default=datetime.utcnow)
    role = Column(Enum(UserRole), default=UserRole.COMMON)

    habits = relationship("UserHabit", back_populates="user")
    profile = relationship("Profile", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")
    interactions = relationship("Interaction", back_populates="user")


class Habit(Base):
    __tablename__ = "Habit"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    category = Column(Enum(HabitCategory))
    description = Column(String)

    users = relationship("UserHabit", back_populates="habit")


class UserHabit(Base):
    __tablename__ = "UserHabit"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String, ForeignKey("User.id"))
    habitId = Column(String, ForeignKey("Habit.id"))
    status = Column(Enum(HabitStatus))
    scheduledTime = Column(String)
    completedAt = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="habits")
    habit = relationship("Habit", back_populates="users")


class Profile(Base):
    __tablename__ = "Profile"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String, ForeignKey("User.id"))
    profileType = Column(Enum(ProfileType))
    assignedAt = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="profile")


class Recommendation(Base):
    __tablename__ = "Recommendation"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String, ForeignKey("User.id"))
    message = Column(String)
    createdAt = Column(DateTime, default=datetime.utcnow)
    shownTime = Column(String)
    
    user = relationship("User", back_populates="recommendations")

class Interaction(Base): 
    __tablename__ = "Interaction"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String, ForeignKey("User.id"))
    type = Column(Enum(InteractionType))
    target = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="interactions")