from agents.coghp import CoGHPAgent
from agents.fm_coghp import FMCoGHPAgent
from agents.hcoghp import HCoGHPAgent
from agents.hrm_action_coghp import HRMActionCoGHPAgent
from agents.navigation_coghp import NavigationCoGHPAgent

agents = dict(
    coghp=CoGHPAgent,
    fm_coghp=FMCoGHPAgent,
    hcoghp=HCoGHPAgent,
    hrm_action_coghp=HRMActionCoGHPAgent,
    navigation_coghp=NavigationCoGHPAgent,
)
