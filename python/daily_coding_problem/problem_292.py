students = {
    0: [3],
    1: [2],
    2: [1, 4],
    3: [0, 4, 5],
    4: [2, 3],
    5: [3]
}


def solve():
    all_students = list(students.keys())
    team = set()
    team.add(all_students[0])
    enemies = set(students[all_students[0]])

    for student in students.keys():

        if student in team:
            continue

        if student not in enemies:
            team.add(student)
            enemies = enemies.union(set(students[student]))

    team2 = set(students.keys()) - team
    team2_enemies = set(s for student in team2 for s in students[student])
    if team2.intersection(team2_enemies):
        print("False")
    else:
        print(team, team2)


if __name__ == "__main__":
    solve()
