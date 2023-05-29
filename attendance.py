import pymysql
from datetime import datetime


# MySQL 서버에 연결
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='cms',
    cursorclass=pymysql.cursors.DictCursor
)

# 커서 생성
cursor = conn.cursor()
def check_attendance(student_id, attend_type, teacher_id="12345"):

    current_date = datetime.now().strftime('%Y-%m-%d')
    sql = "INSERT INTO attendance (attend_type, date, student_id, teacher_id) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (attend_type, current_date, student_id, teacher_id))

    # 변경사항을 커밋
    conn.commit()
# 0 - 출첵, 1 - 결석, 2 - 실종

check_attendance("student1", 0)


def calc_focus(blink):
    reply = QMessageBox.question(
        self, 'Message', 'Are you sure you want to exit?',
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
    )

    if reply == QMessageBox.Yes:
        # 원하는 로직을 여기에 추가합니다.
        print("종료되었습니다.")
        event.accept()
    else:
        event.ignore()

