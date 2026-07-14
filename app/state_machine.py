import logging
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AppState(Enum):
    START = auto()
    INITIALIZING = auto()
    SEARCHING_FOR_HAND = auto()
    READY = auto()
    INTERACTING = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()
    SHUTDOWN = auto()


class StateMachine:
    def __init__(self):
        self._state = AppState.START
        self._transitions: dict = {}
        self._enter_handlers: dict = {}
        self._exit_handlers: dict = {}
        self._state_handlers: dict = {}
        self._no_hand_time: float = 0.0

    @property
    def state(self) -> AppState:
        return self._state

    def on_enter(self, state: AppState, handler: Callable):
        self._enter_handlers[state] = handler

    def on_exit(self, state: AppState, handler: Callable):
        self._exit_handlers[state] = handler

    def on_state(self, state: AppState, handler: Callable):
        self._state_handlers[state] = handler

    def add_transition(self, from_state: AppState, to_state: AppState):
        self._transitions[(from_state, to_state)] = True

    def can_transition(self, to_state: AppState) -> bool:
        return self._transitions.get((self._state, to_state), False)

    def transition(self, to_state: AppState) -> bool:
        if not self.can_transition(to_state):
            logger.warning(f"Invalid transition: {self._state.name} -> {to_state.name}")
            return False

        if self._state in self._exit_handlers:
            self._exit_handlers[self._state]()

        prev = self._state
        self._state = to_state
        logger.info(f"State: {prev.name} -> {to_state.name}")

        if to_state in self._enter_handlers:
            self._enter_handlers[to_state]()
        return True

    def update(self, dt: float):
        if self._state in self._state_handlers:
            self._state_handlers[self._state](dt)

    def force_set(self, state: AppState):
        self._state = state
        logger.info(f"State force set: {state.name}")
        if state in self._enter_handlers:
            self._enter_handlers[state]()


def build_default_state_machine() -> StateMachine:
    sm = StateMachine()

    sm.add_transition(AppState.START, AppState.INITIALIZING)
    sm.add_transition(AppState.INITIALIZING, AppState.SEARCHING_FOR_HAND)
    sm.add_transition(AppState.INITIALIZING, AppState.ERROR)
    sm.add_transition(AppState.SEARCHING_FOR_HAND, AppState.READY)
    sm.add_transition(AppState.SEARCHING_FOR_HAND, AppState.ERROR)
    sm.add_transition(AppState.READY, AppState.INTERACTING)
    sm.add_transition(AppState.READY, AppState.SEARCHING_FOR_HAND)
    sm.add_transition(AppState.INTERACTING, AppState.READY)
    sm.add_transition(AppState.INTERACTING, AppState.SEARCHING_FOR_HAND)
    sm.add_transition(AppState.INTERACTING, AppState.ERROR)
    sm.add_transition(AppState.READY, AppState.ERROR)
    sm.add_transition(AppState.ERROR, AppState.SEARCHING_FOR_HAND)
    sm.add_transition(AppState.ERROR, AppState.SHUTTING_DOWN)
    sm.add_transition(AppState.SHUTTING_DOWN, AppState.SHUTDOWN)

    return sm
