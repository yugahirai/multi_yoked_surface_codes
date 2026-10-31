import numpy as np
from ._format_matrix import format_matrix


def extend_circuit_matrix_with_level_1_checks(
    circuit_matrix: np.ndarray,
    observables: np.ndarray,
    elementary_matrix: np.ndarray,
    elementary_observables: np.ndarray,
    num_l1_checks: int,
    first_check: bool = False,
) -> np.ndarray:
    l1_checks = elementary_matrix[:num_l1_checks]
    assert circuit_matrix is not None
    l1_checks = elementary_matrix[:num_l1_checks]
    num_rows_circuit_matrix = circuit_matrix.shape[0]
    num_cols_circuit_matrix = circuit_matrix.shape[1]
    num_rows_elementary_matrix = elementary_matrix.shape[0]
    num_cols_elementary_matrix = elementary_matrix.shape[1]
    num_rows_l1_checks = l1_checks.shape[0]
    num_cols_l1_checks = l1_checks.shape[1]
    num_rows_l2_checks = elementary_matrix[num_l1_checks:].shape[0]
    num_cols_l2_checks = elementary_matrix[num_l1_checks:].shape[1]
    if first_check:
        new_columns = np.zeros(
            (
                num_rows_circuit_matrix - num_rows_elementary_matrix,
                num_rows_elementary_matrix + num_cols_l1_checks,
            ),
            dtype=int,
        )

        new_columns = np.concatenate(
            [
                new_columns,
                np.concatenate(
                    [
                        np.identity(num_rows_elementary_matrix),
                        np.zeros(
                            (num_rows_elementary_matrix, num_cols_elementary_matrix),
                            dtype=int,
                        ),
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )

        new_rows = np.concatenate(
            [
                np.zeros((num_rows_l1_checks, num_cols_circuit_matrix), dtype=int),
                np.identity(num_rows_l1_checks),
                np.zeros((num_rows_l1_checks, num_rows_l2_checks), dtype=int),
                l1_checks,
            ],
            axis=1,
        )
        circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)

        circuit_matrix = np.concatenate([circuit_matrix, new_rows], axis=0)

        observables = np.concatenate(
            (
                observables,
                np.zeros((1, num_rows_elementary_matrix), dtype=int),
                elementary_observables,
            ),
            axis=1,
        )
    else:
        new_columns = np.zeros(
            (
                num_rows_circuit_matrix - num_rows_l1_checks,
                num_rows_l1_checks + num_cols_l1_checks,
            ),
            dtype=int,
        )
        new_columns = np.concatenate(
            [
                new_columns,
                np.concatenate(
                    [
                        np.identity(num_rows_l1_checks),
                        np.zeros(
                            (num_rows_l1_checks, num_cols_l1_checks),
                            dtype=int,
                        ),
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )
        new_rows = np.concatenate(
            [
                np.zeros((num_rows_l1_checks, num_cols_circuit_matrix), dtype=int),
                np.identity(num_rows_l1_checks),
                l1_checks,
            ],
            axis=1,
        )
        circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)
        circuit_matrix = np.concatenate([circuit_matrix, new_rows], axis=0)

        observables = np.concatenate(
            (
                observables,
                np.zeros((1, num_rows_l1_checks), dtype=int),
                elementary_observables,
            ),
            axis=1,
        )

    return circuit_matrix, observables


def extend_circuit_matrix_with_all(
    circuit_matrix: np.ndarray,
    observables: np.ndarray,
    tick_0: bool,
    elementary_matrix: np.ndarray,
    elementary_observables: np.ndarray,
    num_l1_checks: int,
    l1_check_rate: int = 0,
) -> np.ndarray:
    assert circuit_matrix is not None
    l1_checks = elementary_matrix[:num_l1_checks]
    l2_checks = elementary_matrix[num_l1_checks:]
    num_rows_circuit_matrix = circuit_matrix.shape[0]
    num_cols_circuit_matrix = circuit_matrix.shape[1]
    num_rows_l1_checks = l1_checks.shape[0]
    num_cols_l1_checks = l1_checks.shape[1]
    num_rows_l2_checks = l2_checks.shape[0]
    num_cols_l2_checks = l2_checks.shape[1]
    num_rows_elementary_matrix = elementary_matrix.shape[0]
    num_cols_elementary_matrix = elementary_matrix.shape[1]

    if tick_0:
        new_columns = np.zeros(
            (
                num_rows_circuit_matrix - num_rows_l1_checks,
                num_rows_l1_checks + num_cols_l1_checks,
            ),
            dtype=int,
        )
        new_columns = np.concatenate(
            [
                new_columns,
                np.concatenate(
                    [
                        np.identity(num_rows_l1_checks),
                        np.zeros((num_rows_l1_checks, num_cols_l1_checks), dtype=int),
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )
    else:
        new_columns = np.zeros(
            (
                num_rows_circuit_matrix - num_rows_l1_checks,
                num_rows_l1_checks + num_cols_l1_checks,
            ),
            dtype=int,
        )

        new_columns = np.concatenate(
            [
                new_columns,
                np.concatenate(
                    [
                        np.identity(num_rows_l1_checks),
                        np.zeros(
                            (num_rows_l1_checks, num_cols_l1_checks),
                            dtype=int,
                        ),
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )

    new_rows = np.concatenate(
        [
            np.zeros((num_rows_l1_checks, num_cols_circuit_matrix), dtype=int),
            np.identity(num_rows_l1_checks),
            l1_checks,
        ],
        axis=1,
    )

    if tick_0:
        n_post_init_l1 = max(0, l1_check_rate - 2) + 1
        new_l2_rows = l2_checks.copy()
        for _ in range(n_post_init_l1):
            new_l2_rows = np.concatenate(
                (
                    new_l2_rows,
                    np.zeros((num_rows_l2_checks, num_rows_l1_checks), dtype=int),
                    l2_checks,
                ),
                axis=1,
            )

        observables = np.concatenate(
            (
                observables,
                np.zeros((1, num_rows_l1_checks), dtype=int),
                elementary_observables,
            ),
            axis=1,
        )
    else:
        n_mid_l1 = max(0, l1_check_rate - 2)
        new_l2_rows = np.concatenate(
            (
                np.zeros(
                    (
                        num_rows_l2_checks,
                        num_cols_circuit_matrix
                        - num_cols_l1_checks
                        - num_rows_elementary_matrix
                        - n_mid_l1 * num_rows_l1_checks
                        - n_mid_l1 * num_cols_l1_checks,
                    ),
                    dtype=int,
                ),
                np.zeros((num_rows_l2_checks, num_rows_l1_checks), dtype=int),
                np.identity((num_rows_l2_checks), dtype=int),
            ),
            axis=1,
        )
        for _ in range(n_mid_l1 + 1):
            new_l2_rows = np.concatenate(
                (
                    new_l2_rows,
                    l2_checks,
                    np.zeros((num_rows_l2_checks, num_rows_l1_checks), dtype=int),
                ),
                axis=1,
            )
        new_l2_rows = np.concatenate(
            (
                new_l2_rows,
                l2_checks,
            ),
            axis=1,
        )
        observables = np.concatenate(
            (
                observables,
                np.zeros((1, num_rows_l1_checks), dtype=int),
                elementary_observables,
            ),
            axis=1,
        )

    circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)
    circuit_matrix = np.concatenate([circuit_matrix, new_rows], axis=0)
    circuit_matrix = np.concatenate([circuit_matrix, new_l2_rows], axis=0)
    return circuit_matrix, observables


def extend_circuit_matrix_with_last_check(
    circuit_matrix: np.ndarray,
    observables: np.ndarray,
    elementary_matrix: np.ndarray,
    elementary_observables: np.ndarray,
    num_l1_checks: int,
) -> np.ndarray:
    assert circuit_matrix is not None
    l1_checks = elementary_matrix[:num_l1_checks]
    l2_checks = elementary_matrix[num_l1_checks:]
    num_rows_circuit_matrix = circuit_matrix.shape[0]
    num_cols_circuit_matrix = circuit_matrix.shape[1]
    num_rows_l1_checks = l1_checks.shape[0]
    num_cols_l1_checks = l1_checks.shape[1]
    num_rows_l2_checks = l2_checks.shape[0]
    num_cols_l2_checks = l2_checks.shape[1]
    num_rows_elementary_matrix = elementary_matrix.shape[0]
    num_cols_elementary_matrix = elementary_matrix.shape[1]

    new_columns = np.zeros(
        (
            num_rows_circuit_matrix - num_rows_elementary_matrix,
            num_rows_elementary_matrix + num_cols_l1_checks,
        ),
        dtype=int,
    )
    new_columns = np.concatenate(
        [
            new_columns,
            np.concatenate(
                [
                    np.identity(num_rows_elementary_matrix),
                    np.zeros(
                        (num_rows_elementary_matrix, num_cols_l1_checks), dtype=int
                    ),
                ],
                axis=1,
            ),
        ],
        axis=0,
    )

    new_rows = np.concatenate(
        [
            np.zeros((num_rows_elementary_matrix, num_cols_circuit_matrix), dtype=int),
            np.identity(num_rows_elementary_matrix),
            elementary_matrix,
        ],
        axis=1,
    )

    circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)
    circuit_matrix = np.concatenate([circuit_matrix, new_rows], axis=0)

    observables = np.concatenate(
        (
            observables,
            np.zeros((1, num_rows_elementary_matrix), dtype=int),
            elementary_observables,
        ),
        axis=1,
    )
    return circuit_matrix, observables


def extend_circuit_matrix_with_only_l1(
    circuit_matrix: np.ndarray,
    observables: np.ndarray,
    elementary_matrix: np.ndarray,
    elementary_observables: np.ndarray,
    num_l1_checks: int,
) -> np.ndarray:
    assert circuit_matrix is not None
    l1_checks = elementary_matrix
    num_rows_circuit_matrix = circuit_matrix.shape[0]
    num_cols_circuit_matrix = circuit_matrix.shape[1]
    num_rows_l1_checks = l1_checks.shape[0]
    num_cols_l1_checks = l1_checks.shape[1]

    if num_rows_circuit_matrix > 1:
        new_columns = np.zeros(
            (
                num_rows_circuit_matrix - 1,
                num_rows_l1_checks + num_cols_l1_checks,
            ),
            dtype=int,
        )
        a = np.concatenate(
            [
                np.identity(num_rows_l1_checks),
                np.zeros((num_rows_l1_checks, num_cols_l1_checks), dtype=int),
            ],
            axis=1,
        )
        new_columns = np.concatenate([new_columns, a], axis=0)
    else:
        new_columns = np.identity(num_rows_l1_checks)
        new_columns = np.concatenate(
            [
                new_columns,
                np.zeros((num_rows_l1_checks, num_cols_l1_checks), dtype=int),
            ],
            axis=1,
        )
    new_row = np.concatenate(
        [
            np.zeros((num_rows_l1_checks, num_cols_circuit_matrix), dtype=int),
            np.identity(num_rows_l1_checks),
            l1_checks,
        ],
        axis=1,
    )
    circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)
    circuit_matrix = np.concatenate([circuit_matrix, new_row], axis=0)

    observables = np.concatenate(
        (
            observables,
            np.zeros((1, num_rows_l1_checks), dtype=int),
            elementary_observables,
        ),
        axis=1,
    )
    return circuit_matrix, observables


def extend_circuit_matrix_without_measure_error(
    circuit_matrix: np.ndarray,
    observables: np.ndarray,
    elementary_matrix: np.ndarray,
    elementary_observables: np.ndarray,
) -> np.ndarray:
    elementary_matrix = np.asarray(elementary_matrix, dtype=int)
    elementary_observables = np.asarray(elementary_observables, dtype=int)

    if circuit_matrix is None:
        circuit_matrix = elementary_matrix
        observables = elementary_observables
        return circuit_matrix, observables
    else:
        circuit_matrix = np.asarray(circuit_matrix, dtype=int)
        observables = np.asarray(observables, dtype=int)
        num_rows_circuit_matrix = circuit_matrix.shape[0]
        num_cols_circuit_matrix = circuit_matrix.shape[1]
        num_rows_elementary_matrix = elementary_matrix.shape[0]
        num_cols_elementary_matrix = elementary_matrix.shape[1]
        new_columns = np.zeros(
            (
                num_rows_circuit_matrix,
                num_cols_elementary_matrix,
            ),
            dtype=int,
        )
        new_row = np.concatenate(
            [
                np.zeros(
                    (num_rows_elementary_matrix, num_cols_circuit_matrix), dtype=int
                ),
                elementary_matrix,
            ],
            axis=1,
        )
        circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)
        circuit_matrix = np.concatenate([circuit_matrix, new_row], axis=0)

        observables = np.concatenate(
            (
                observables,
                elementary_observables,
            ),
            axis=1,
        )

        return circuit_matrix, observables


def extend_circuit_matrix_with_all_without_measure_error(
    circuit_matrix: np.ndarray,
    observables: np.ndarray,
    l1_check_rate: int,
    elementary_matrix: np.ndarray,
    elementary_observables: np.ndarray,
    l1_elementary_matrix: np.ndarray,
    l2_elementary_matrix: np.ndarray,
) -> np.ndarray:
    elementary_matrix = np.asarray(elementary_matrix, dtype=int)
    elementary_observables = np.asarray(elementary_observables, dtype=int)
    l1_elementary_matrix = np.asarray(l1_elementary_matrix, dtype=int)
    l2_elementary_matrix = np.asarray(l2_elementary_matrix, dtype=int)

    if circuit_matrix is None:
        assert l1_check_rate == 1
        return elementary_matrix, elementary_observables

    num_rows_circuit_matrix = circuit_matrix.shape[0]
    num_cols_circuit_matrix = circuit_matrix.shape[1]
    num_rows_l1_checks = l1_elementary_matrix.shape[0]
    num_cols_l1_checks = l1_elementary_matrix.shape[1]
    num_rows_l2_checks = l2_elementary_matrix.shape[0]
    num_cols_l2_checks = l2_elementary_matrix.shape[1]

    new_columns = np.zeros(
        (
            num_rows_circuit_matrix,
            num_cols_l1_checks,
        ),
        dtype=int,
    )

    new_row = np.concatenate(
        [
            np.zeros((num_rows_l1_checks, num_cols_circuit_matrix), dtype=int),
            l1_elementary_matrix,
        ],
        axis=1,
    )

    l2_rows = np.zeros(
        (
            num_rows_l2_checks,
            num_cols_circuit_matrix - (l1_check_rate - 1) * num_cols_l1_checks,
        ),
        dtype=int,
    )
    l2_rows = np.concatenate(
        [
            l2_rows,
            np.tile(l2_elementary_matrix, (1, l1_check_rate)),
        ],
        axis=1,
    )
    new_row = np.concatenate([new_row, l2_rows], axis=0)
    circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)
    circuit_matrix = np.concatenate([circuit_matrix, new_row], axis=0)

    observables = np.concatenate(
        (
            observables,
            elementary_observables,
        ),
        axis=1,
    )

    return circuit_matrix, observables


def extend_circuit_matrix_with_all_without_measure_error_for_three_yoke(
    circuit_matrix: np.ndarray,
    observables: np.ndarray,
    l1_check_rate: int,
    l2_check_rate: int,
    elementary_matrix: np.ndarray,
    elementary_observables: np.ndarray,
    l1_elementary_matrix: np.ndarray,
    l2_elementary_matrix: np.ndarray,
    l3_elementary_matrix: np.ndarray,
) -> np.ndarray:
    assert circuit_matrix is not None
    elementary_matrix = np.asarray(elementary_matrix, dtype=int)
    elementary_observables = np.asarray(elementary_observables, dtype=int)
    l1_elementary_matrix = np.asarray(l1_elementary_matrix, dtype=int)
    l2_elementary_matrix = np.asarray(l2_elementary_matrix, dtype=int)
    l3_elementary_matrix = np.asarray(l3_elementary_matrix, dtype=int)

    num_rows_circuit_matrix = circuit_matrix.shape[0]
    num_cols_circuit_matrix = circuit_matrix.shape[1]
    num_rows_l1_checks = l1_elementary_matrix.shape[0]
    num_cols_l1_checks = l1_elementary_matrix.shape[1]
    num_rows_l2_checks = l2_elementary_matrix.shape[0]
    num_cols_l2_checks = l2_elementary_matrix.shape[1]
    num_rows_l3_checks = l3_elementary_matrix.shape[0]
    num_cols_l3_checks = l3_elementary_matrix.shape[1]

    new_columns = np.zeros(
        (
            num_rows_circuit_matrix,
            num_cols_l1_checks,
        ),
        dtype=int,
    )

    new_row = np.concatenate(
        [
            np.zeros((num_rows_l1_checks, num_cols_circuit_matrix), dtype=int),
            l1_elementary_matrix,
        ],
        axis=1,
    )

    l3_rows = np.zeros(
        (
            num_rows_l3_checks,
            num_cols_circuit_matrix - (l1_check_rate - 1) * num_cols_l1_checks,
        ),
        dtype=int,
    )
    l3_rows = np.concatenate(
        [
            l3_rows,
            np.tile(l3_elementary_matrix, (1, l1_check_rate)),
        ],
        axis=1,
    )

    l2_rows = np.zeros(
        (
            num_rows_l2_checks,
            num_cols_circuit_matrix
            - (l1_check_rate // l2_check_rate - 1) * num_cols_l1_checks,
        ),
        dtype=int,
    )
    l2_rows = np.concatenate(
        [
            l2_rows,
            np.tile(l2_elementary_matrix, (1, l1_check_rate // l2_check_rate)),
        ],
        axis=1,
    )

    new_row = np.concatenate([new_row, l2_rows, l3_rows], axis=0)
    circuit_matrix = np.concatenate([circuit_matrix, new_columns], axis=1)
    circuit_matrix = np.concatenate([circuit_matrix, new_row], axis=0)

    observables = np.concatenate(
        (
            observables,
            elementary_observables,
        ),
        axis=1,
    )

    return circuit_matrix, observables
