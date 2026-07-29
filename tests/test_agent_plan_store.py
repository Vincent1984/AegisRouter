"""AgentPlanStore 单元测试

验证检查点 V1-1: AgentPlanStore 基本 set/get 操作正确

测试内容:
- TC-STORE-001: 基本 set/get 操作
- 多次 set/get 操作正确性
- 重复 set 同一 key 覆盖旧值
"""

import pytest

from aegis_router.router.agent_plan_store import AgentPlanStore


class TestAgentPlanStoreBasicSetGet:
    """V1-1: AgentPlanStore 基本 set/get 操作正确"""

    def test_set_then_get_returns_correct_value(self):
        """set_model('agent_a', 'model_x') 后 get_model('agent_a') 返回 'model_x'"""
        store = AgentPlanStore()
        store.set_model("agent_a", "model_x")
        assert store.get_model("agent_a") == "model_x"

    def test_multiple_set_get_operations(self):
        """多个 agent 的 set/get 操作互不干扰"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        store.set_model("agent_b", "claude-3")
        store.set_model("agent_c", "deepseek-v2")

        assert store.get_model("agent_a") == "gpt-4"
        assert store.get_model("agent_b") == "claude-3"
        assert store.get_model("agent_c") == "deepseek-v2"

    def test_overwrite_key_updates_value(self):
        """重复 set 同一 key 覆盖旧值"""
        store = AgentPlanStore()
        store.set_model("agent_a", "model_old")
        store.set_model("agent_a", "model_new")

        assert store.get_model("agent_a") == "model_new"

    def test_overwrite_does_not_change_length(self):
        """覆盖同一 key 后长度不变"""
        store = AgentPlanStore()
        store.set_model("agent_a", "model_old")
        store.set_model("agent_a", "model_new")

        assert len(store) == 1

    def test_set_get_consistency_across_operations(self):
        """连续操作后所有键值保持一致"""
        store = AgentPlanStore()

        # 初始设置
        store.set_model("intent_classifier", "gpt-4o-mini")
        store.set_model("reasoning_engine", "gpt-4o")
        store.set_model("code_assistant", "claude-3-sonnet")

        # 覆盖其中一个
        store.set_model("reasoning_engine", "o1-preview")

        # 验证所有值
        assert store.get_model("intent_classifier") == "gpt-4o-mini"
        assert store.get_model("reasoning_engine") == "o1-preview"
        assert store.get_model("code_assistant") == "claude-3-sonnet"
        assert len(store) == 3


class TestAgentPlanStoreGetUnknown:
    """TC-STORE-002: get_model 对未知 agent 返回 None"""

    def test_get_model_unknown_agent_returns_none(self):
        """空 store 查询任意 agent 返回 None"""
        store = AgentPlanStore()
        assert store.get_model("nonexistent_agent") is None

    def test_get_model_unknown_after_other_agents_set(self):
        """已有其他 agent 时查询未设置的 agent 返回 None"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        store.set_model("agent_b", "claude-3")
        assert store.get_model("agent_c") is None


class TestAgentPlanStoreContains:
    """TC-STORE-003: __contains__ 正确判断"""

    def test_contains_existing_agent(self):
        """已设置的 agent 在 store 中"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        assert "agent_a" in store

    def test_not_contains_unknown_agent(self):
        """未设置的 agent 不在 store 中"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        assert "agent_b" not in store

    def test_contains_empty_store(self):
        """空 store 不包含任何 agent"""
        store = AgentPlanStore()
        assert "agent_a" not in store


class TestAgentPlanStoreLen:
    """TC-STORE-004: __len__ 返回正确数量"""

    def test_len_empty_store(self):
        """空 store 长度为 0"""
        store = AgentPlanStore()
        assert len(store) == 0

    def test_len_after_single_set(self):
        """设置一个 agent 后长度为 1"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        assert len(store) == 1

    def test_len_after_multiple_sets(self):
        """设置多个不同 agent 后长度正确"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        store.set_model("agent_b", "claude-3")
        store.set_model("agent_c", "deepseek-v2")
        assert len(store) == 3


class TestAgentPlanStoreGetAllPlans:
    """TC-STORE-005: get_all_plans 返回完整映射"""

    def test_get_all_plans_empty(self):
        """空 store 返回空字典"""
        store = AgentPlanStore()
        assert store.get_all_plans() == {}

    def test_get_all_plans_returns_complete_mapping(self):
        """返回所有已设置的 agent → model 映射"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        store.set_model("agent_b", "claude-3")
        store.set_model("agent_c", "deepseek-v2")

        plans = store.get_all_plans()
        assert plans == {
            "agent_a": "gpt-4",
            "agent_b": "claude-3",
            "agent_c": "deepseek-v2",
        }

    def test_get_all_plans_returns_copy(self):
        """get_all_plans 返回副本，修改不影响内部状态"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")

        plans = store.get_all_plans()
        plans["agent_a"] = "tampered"

        assert store.get_model("agent_a") == "gpt-4"


class TestAgentPlanStoreOverwrite:
    """TC-STORE-006: 重复 set 同一 key 覆盖旧值，长度不变"""

    def test_overwrite_replaces_value(self):
        """覆盖后 get_model 返回新值"""
        store = AgentPlanStore()
        store.set_model("agent_a", "old_model")
        store.set_model("agent_a", "new_model")
        assert store.get_model("agent_a") == "new_model"

    def test_overwrite_length_unchanged(self):
        """覆盖后长度不增加"""
        store = AgentPlanStore()
        store.set_model("agent_a", "model_v1")
        store.set_model("agent_b", "model_v1")
        assert len(store) == 2

        store.set_model("agent_a", "model_v2")
        assert len(store) == 2

    def test_overwrite_does_not_affect_other_keys(self):
        """覆盖一个 key 不影响其他 key 的值"""
        store = AgentPlanStore()
        store.set_model("agent_a", "gpt-4")
        store.set_model("agent_b", "claude-3")

        store.set_model("agent_a", "gpt-4o")

        assert store.get_model("agent_a") == "gpt-4o"
        assert store.get_model("agent_b") == "claude-3"
