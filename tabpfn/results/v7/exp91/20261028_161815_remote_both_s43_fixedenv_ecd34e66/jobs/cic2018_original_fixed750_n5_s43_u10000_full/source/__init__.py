"""Runnable prototype APIs; see run_designs.py for the registered experiment CLI."""
from .common import TabPFNFactory
from .teacher_student import StudentConfig, TeacherStudentClassifier
from .pair_residual import PairConfig, PairResidualClassifier

__all__ = ['TabPFNFactory', 'StudentConfig', 'TeacherStudentClassifier',
           'PairConfig', 'PairResidualClassifier']
