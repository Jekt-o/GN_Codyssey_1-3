"""표준 라이브러리만으로 구현한 Mini NPU 콘솔 시뮬레이터 (Python 3.8+)."""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple


EPSILON = 1e-9
DEFAULT_REPEATS = 1000
Matrix = List[List[float]]
FilterBank = Dict[str, Dict[str, Matrix]]


def normalize_label(value: object) -> str:
    """외부 라벨을 내부 표준 라벨 Cross 또는 X로 바꾼다."""
    if isinstance(value, str):
        label = value.strip().lower()
        if label in ("+", "cross"):
            return "Cross"
        if label == "x":
            return "X"
    raise ValueError("알 수 없는 라벨 {!r}: +/cross/Cross 또는 x/X가 필요합니다.".format(value))


def validate_matrix(value: object, size: int, name: str = "행렬") -> Matrix:
    """JSON 행렬의 크기와 유한한 숫자를 검증하고 float 행렬을 반환한다."""
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("{}: 행 수는 {}개여야 합니다.".format(name, size))
    matrix = []
    for row_index, row in enumerate(value, 1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("{} {}행: 열 수는 {}개여야 합니다.".format(name, row_index, size))
        converted = []
        for column_index, item in enumerate(row, 1):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("{} ({}, {}): 숫자가 필요합니다.".format(name, row_index, column_index))
            try:
                number = float(item)
            except (ValueError, OverflowError):
                raise ValueError("{}: float로 표현할 수 없는 숫자입니다.".format(name)) from None
            if not math.isfinite(number):
                raise ValueError("{}: NaN/Infinity는 사용할 수 없습니다.".format(name))
            converted.append(number)
        matrix.append(converted)
    return matrix


def mac(pattern: Matrix, kernel: Matrix) -> float:
    """검증된 같은 크기 행렬에 대해 위치별 곱셈과 누적 덧셈을 수행한다."""
    score = 0.0
    size = len(pattern)
    for row in range(size):
        for column in range(size):
            score += pattern[row][column] * kernel[row][column]
    if not math.isfinite(score):
        raise ValueError("MAC 점수가 유한하지 않습니다. 입력값의 크기를 줄이세요.")
    return score


def decide(score_a: float, score_b: float, label_a: str = "Cross", label_b: str = "X") -> str:
    if not math.isfinite(score_a) or not math.isfinite(score_b):
        raise ValueError("유한한 점수만 비교할 수 있습니다.")
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def generate_patterns(size: int) -> Dict[str, Matrix]:
    """중앙 행·열의 십자가와 두 대각선의 X를 생성한다."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("패턴 생성 크기는 양의 정수여야 합니다.")
    center = {(size - 1) // 2, size // 2}
    cross = [[float(row in center or column in center) for column in range(size)]
             for row in range(size)]
    diagonal = [[float(row == column or row + column == size - 1) for column in range(size)]
                for row in range(size)]
    return {"Cross": cross, "X": diagonal}


def read_matrix(name: str, size: int = 3, *, allow_separator: bool = False) -> Matrix:
    print("\n{} ({}줄, 각 줄에 숫자 {}개를 공백으로 구분)".format(name, size, size))
    print("입력 도중 .을 입력하면 처음부터 다시 입력합니다. 첫 줄에 cross 또는 x를 입력하면 자동 생성합니다.")
    matrix = []
    while len(matrix) < size:
        text = input("  {}행: ".format(len(matrix) + 1)).strip()
        if allow_separator:
            # 이전 입력 형식의 행렬 사이 구분자를 한 번 건너뛴다.
            allow_separator = False
            if text == ".":
                continue
        if text == ".":
            print("입력 형식 오류: 행 수는 {}개여야 합니다 (입력: {}개). 처음부터 다시 입력하세요.".format(
                size, len(matrix)))
            matrix.clear()
            continue
        if not matrix and text.lower() in ("cross", "x"):
            label = normalize_label(text)
            print("{}: {} 패턴을 생성했습니다.".format(name, label))
            return generate_patterns(size)[label]
        try:
            parts = text.split()
            if len(parts) != size:
                raise ValueError
            row = [float(part) for part in parts]
            if not all(math.isfinite(number) for number in row):
                raise ValueError
        except (ValueError, OverflowError):
            print("입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요.".format(size))
            continue
        matrix.append(row)
    return matrix


def mac_flat(pattern: List[float], kernel: List[float]) -> float:
    """같은 크기의 1차원 배열에 대해 위치별 곱셈과 누적 덧셈을 수행한다."""
    score = 0.0
    for index in range(len(pattern)):
        score += pattern[index] * kernel[index]
    if not math.isfinite(score):
        raise ValueError("MAC 점수가 유한하지 않습니다. 입력값의 크기를 줄이세요.")
    return score


def benchmark_mac(pattern: Matrix, kernel: Matrix, repeats: int) -> Tuple[float, float]:
    """동일한 입력에 대한 2차원과 1차원 MAC의 평균 시간(ms)을 구한다."""
    if repeats < 10:
        raise ValueError("성능 측정은 최소 10회 반복해야 합니다.")
    flat_pattern = [value for row in pattern for value in row]
    flat_kernel = [value for row in kernel for value in row]
    mac(pattern, kernel)
    mac_flat(flat_pattern, flat_kernel)
    elapsed_2d = elapsed_1d = 0.0
    for index in range(repeats):
        if index % 2 == 0:
            start = perf_counter()
            mac(pattern, kernel)
            middle = perf_counter()
            mac_flat(flat_pattern, flat_kernel)
            end = perf_counter()
            elapsed_2d += middle - start
            elapsed_1d += end - middle
        else:
            start = perf_counter()
            mac_flat(flat_pattern, flat_kernel)
            middle = perf_counter()
            mac(pattern, kernel)
            end = perf_counter()
            elapsed_1d += middle - start
            elapsed_2d += end - middle
    return elapsed_2d * 1000 / repeats, elapsed_1d * 1000 / repeats


def print_performance(sizes: List[int], repeats: int,
                      manual_pair: Optional[Tuple[Matrix, Matrix]] = None) -> None:
    print("\n[성능 분석: MAC 호출당 평균 / {}회]".format(repeats))
    print("크기(N×N)     2D 평균(ms)     1D 평균(ms)     MAC 횟수(N²)")
    print("-----------------------------------------------------------")
    for size in sizes:
        if manual_pair is not None:
            pattern, kernel = manual_pair
        else:
            generated = generate_patterns(size)
            pattern, kernel = generated["Cross"], generated["Cross"]
        average_2d, average_1d = benchmark_mac(pattern, kernel, repeats)
        print("{:<12} {:>11.6f} {:>15.6f} {:>15}".format(
            "{}×{}".format(size, size), average_2d, average_1d, size * size))
    print("필터 1개 기준: N²번 곱셈·누적 덧셈. 두 필터 판정에는 2N²번이 필요합니다.")
    print("동일 입력과 반복 횟수로 측정했습니다. 1차원 변환 시간은 제외했습니다.")
    if manual_pair is None:
        print("측정 입력: 크기별로 생성한 Cross 패턴과 필터 (생성 시간 제외).")
    else:
        print("측정 입력: 입력한 패턴과 필터 A.")


def run_manual(repeats: int) -> int:
    filter_a = read_matrix("필터 A")
    print("필터 A 저장 완료")
    filter_b = read_matrix("필터 B", allow_separator=True)
    print("필터 B 저장 완료")
    while True:
        pattern = read_matrix("패턴", allow_separator=True)
        try:
            start = perf_counter()
            score_a = mac(pattern, filter_a)
            score_b = mac(pattern, filter_b)
            elapsed_ms = (perf_counter() - start) * 1000
        except ValueError as error:
            print("연산 오류: {} 패턴을 다시 입력하세요.".format(error))
            continue
        break
    prediction = decide(score_a, score_b, "A", "B")
    print("\n[MAC 결과]")
    print("A 점수: {:.17g}".format(score_a))
    print("B 점수: {:.17g}".format(score_b))
    print("두 필터 MAC 연산 시간(1회): {:.6f} ms".format(elapsed_ms))
    if prediction == "UNDECIDED":
        print("판정: 판정 불가 (UNDECIDED, |A-B| < {})".format(EPSILON))
    else:
        print("판정: {}".format(prediction))
    print_performance([3], repeats, (pattern, filter_a))
    return 0


def load_filter_bank(raw_filters: object) -> Tuple[FilterBank, Dict[str, str], Optional[str]]:
    bank = {}  # type: FilterBank
    errors = {}  # type: Dict[str, str]
    if not isinstance(raw_filters, dict):
        return bank, errors, "filters는 size_N 키를 가진 객체여야 합니다."
    for size_key, raw_pair in raw_filters.items():
        try:
            match = re.fullmatch(r"size_([1-9][0-9]*)", size_key)
            if match is None:
                raise ValueError("필터 키는 size_N 형식이어야 합니다.")
            size = int(match.group(1))
            if not isinstance(raw_pair, dict):
                raise ValueError("필터 묶음은 cross와 x 키를 가진 객체여야 합니다.")
            pair = {}
            for label, matrix in raw_pair.items():
                normalized = normalize_label(label)
                if normalized in pair:
                    raise ValueError("정규화 후 중복되는 필터 라벨: {}".format(normalized))
                pair[normalized] = validate_matrix(matrix, size, "{} {} 필터".format(size_key, normalized))
            if set(pair) != {"Cross", "X"}:
                raise ValueError("Cross와 X 필터가 모두 필요합니다.")
            bank[size_key] = pair
        except ValueError as error:
            errors[size_key] = str(error)
    return bank, errors, None


@dataclass
class CaseResult:
    case_id: str
    cross_score: Optional[float] = None
    x_score: Optional[float] = None
    predicted: Optional[str] = None
    expected: Optional[str] = None
    passed: bool = False
    reason: str = ""


def analyze_case(case_id: str, raw_case: object, bank: FilterBank,
                 filter_errors: Dict[str, str], global_filter_error: Optional[str] = None) -> CaseResult:
    result = CaseResult(case_id)
    try:
        match = re.fullmatch(r"size_([1-9][0-9]*)_([0-9]+)", case_id)
        if match is None:
            raise ValueError("패턴 키는 size_N_idx 형식이어야 합니다.")
        size = int(match.group(1))
        if not isinstance(raw_case, dict):
            raise ValueError("패턴 항목은 input과 expected를 가진 객체여야 합니다.")
        result.expected = normalize_label(raw_case.get("expected"))
        pattern = validate_matrix(raw_case.get("input"), size, "입력 패턴")
        size_key = "size_{}".format(size)
        if global_filter_error is not None:
            raise ValueError(global_filter_error)
        if size_key in filter_errors:
            raise ValueError(filter_errors[size_key])
        if size_key not in bank:
            raise ValueError("{} 필터가 없습니다.".format(size_key))
        pair = bank[size_key]
    except ValueError as error:
        result.reason = "[데이터/스키마] {}".format(error)
        return result
    try:
        result.cross_score = mac(pattern, pair["Cross"])
        result.x_score = mac(pattern, pair["X"])
        result.predicted = decide(result.cross_score, result.x_score)
    except ValueError as error:
        result.reason = "[수치 연산] {}".format(error)
        return result
    result.passed = result.predicted == result.expected
    if result.predicted == "UNDECIDED":
        result.reason = "[판정 정책] 동점(|Cross-X| < {}): expected {}와 일치하지 않습니다.".format(
            EPSILON, result.expected)
    elif not result.passed:
        result.reason = "[판정 불일치] 계산 결과 {}, expected {}.".format(result.predicted, result.expected)
    return result


def print_case(result: CaseResult) -> None:
    print("\n--- {} ---".format(result.case_id))
    for label, score in (("Cross", result.cross_score), ("X", result.x_score)):
        print("{} 점수: {}".format(label, "계산 불가" if score is None else "{:.17g}".format(score)))
    print("판정: {} | expected: {} | {}".format(
        result.predicted or "계산 불가", result.expected or "검증 실패", "PASS" if result.passed else "FAIL"))
    if result.reason:
        print("사유: {}".format(result.reason))


def print_summary(results: List[CaseResult]) -> None:
    failures = [result for result in results if not result.passed]
    print("\n[결과 요약]")
    print("총 테스트: {}개".format(len(results)))
    print("통과: {}개".format(len(results) - len(failures)))
    print("실패: {}개".format(len(failures)))
    if failures:
        print("실패 케이스:")
        for result in failures:
            print("- {}: {}".format(result.case_id, result.reason))
    else:
        print("실패 케이스: 없음")


def run_json(path: Path, repeats: int) -> int:
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            document = json.load(source)
        if not isinstance(document, dict):
            raise ValueError("JSON 최상위 값은 객체여야 합니다.")
        patterns = document.get("patterns")
        if not isinstance(patterns, dict) or not patterns:
            raise ValueError("patterns는 하나 이상의 케이스를 가진 객체여야 합니다.")
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print("데이터 파일 오류 ({}): {}".format(path, error))
        print("분석할 수 없어 테스트를 수행하지 않았습니다.")
        return 2

    print("\n[필터 로드]")
    bank, filter_errors, global_error = load_filter_bank(document.get("filters"))
    for size_key in bank:
        print("{} 필터 로드 완료 (Cross, X)".format(size_key))
    for size_key, error in filter_errors.items():
        print("{} 필터 오류: {}".format(size_key, error))
    if global_error:
        print("필터 오류: {}".format(global_error))

    print("\n[패턴 분석]")
    results = []
    for case_id, raw_case in patterns.items():
        result = analyze_case(case_id, raw_case, bank, filter_errors, global_error)
        results.append(result)
        print_case(result)
    print_performance([3, 5, 13, 25], repeats)
    print_summary(results)
    return 0 if all(result.passed for result in results) else 1


def repeat_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("반복 횟수는 정수여야 합니다.") from None
    if count < 10:
        raise argparse.ArgumentTypeError("반복 횟수는 최소 10이어야 합니다.")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="반복문 기반 Mini NPU 시뮬레이터")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--mode", choices=("1", "2"), help="1: 사용자 입력, 2: JSON 분석 (생략하면 메뉴)")
    action.add_argument("--generate", type=int, metavar="N", help="N×N 십자가와 X 패턴 출력")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().with_name("data.json"),
                        help="JSON 파일 경로 (기본: main.py와 같은 폴더의 data.json)")
    parser.add_argument("--repeats", type=repeat_count, default=DEFAULT_REPEATS,
                        help="성능 측정 반복 횟수 (기본: 1000, 최소: 10)")
    arguments = parser.parse_args()
    if arguments.generate is not None and arguments.generate < 1:
        parser.error("패턴 생성 크기는 양의 정수여야 합니다.")
    print("=== Mini NPU Simulator ===")
    try:
        if arguments.generate is not None:
            for label, matrix in generate_patterns(arguments.generate).items():
                print("\n[{}]".format(label))
                for row in matrix:
                    print(" ".join(format(value, "g") for value in row))
            return 0
        mode = arguments.mode
        while mode not in ("1", "2"):
            print("\n[모드 선택]\n1. 사용자 입력 (3×3)\n2. data.json 분석")
            mode = input("선택: ").strip()
            if mode not in ("1", "2"):
                print("1 또는 2를 입력하세요.")
        if mode == "1":
            return run_manual(arguments.repeats)
        return run_json(arguments.data, arguments.repeats)
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되거나 취소되어 프로그램을 종료합니다.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
