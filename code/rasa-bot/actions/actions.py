# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import FollowupAction
from rasa_sdk.events import SlotSet
import numpy as np
import pandas as pd


class SetHints(Action):
    def name(self):
        return "action_set_want_hints"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],) -> List[Dict[Text, Any]]:

        intent = str(tracker.latest_message['intent'].get('name'))

        if intent == 'hints':
            return [SlotSet("want_hints", True)]

        else:
            return [SlotSet("want_hints", False)]


class PickActionSuggestive(Action):
    """Picks a random action when it predicts the intent suggestive.
    Cooperativeness is the probability that the child will confirm what the
    interviewer suggested.
    Else, it will with equal probability either deny, stay silent, reluct,
    be confused or correct the interviewer."""

    def name(self):
        return "action_pick_action_suggestive"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],) -> List[Dict[Text, Any]]:

        cooperativeness = 0.5

        r = np.random.random(1)
        if r <= cooperativeness:
            if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                return [FollowupAction(name='utter_confirmation'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_suggestive')]
            else:
                return [FollowupAction(name='utter_confirmation')]
        else:
            r2 = np.random.random(1)
            if r2 <= 0.2:
                if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                    return [FollowupAction(name='utter_denial'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_suggestive')]
                else:
                    return [FollowupAction(name='utter_denial')]
            elif r2 <= 0.4:
                if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                    return [FollowupAction(name='utter_silence'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_suggestive')]
                else:
                    return [FollowupAction(name='utter_silence')]
            elif r2 <= 0.6:
                if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                    return [FollowupAction(name='utter_reluctance'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_suggestive')]
                else:
                    return [FollowupAction(name='utter_reluctance')]
            elif r2 <= 0.8:
                if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                    return [FollowupAction(name='utter_confusion'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_suggestive')]
                else:
                    return [FollowupAction(name='utter_confusion')]
            else:
                if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                    return [FollowupAction(name='utter_correction'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_suggestive')]
                else:
                    return [FollowupAction(name='utter_correction')]


class PickActionIntro(Action):
    """ The action is run when the model predicts the action invitation.
    If the entity name_child is detected in the invitation, that is if one of the
    names of the children are mentioned, then the introduction corresponding
    to that child is uttered.
    If no entity is extracted, the action randomly picks which child's storyline
    should be followed. With equal probability, it will utter the
    introduction of one of the children.
    It will also update the slot child which will indicate which child
    we are interviewing.

    Note that if the entity name_child is detected at an earlier point in the
    conversation, the action will still pick a child at random."""

    def name(self):
        return "action_pick_action_intro"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],) -> List[Dict[Text, Any]]:

        # imports the child_replies data
        data = pd.read_csv('child_replies.csv')

        # dictionary for child_names and slot child
        dict_names = {}
        names = list(data.loc[0])[1:]
        values = list(data.columns)[1:]
        for i in range(len(names)):
            dict_names[names[i]] = values[i]

        # finds the detected entity name_child
        person = next(tracker.get_latest_entity_values('name_child'), None)

        # if the entity is detected
        if person != None:
            child = dict_names[person]
            message = list(data[child])[2]
            dispatcher.utter_message(text=str(message))

        else:
            r = np.random.random(1)
            n = len(dict_names)
            prob = np.linspace(0, 1, n+1)
            for i in range(n):
                if r <= prob[i+1]:
                    child = values[i]
                    break
            message = list(data[child])[2]
            dispatcher.utter_message(text=str(message))

        # hint
        if next(tracker.get_latest_entity_values('want_hints'), None) is True:
            # imports the hints
            hints = pd.read_csv('suggestions_replies.csv')
            hint = list(hints[child])[2]
            dispatcher.utter_message(
                text='Here comes a hint for what you can say next:')
            dispatcher.utter_message(text=str(hint))
        return [SlotSet("child", child)]


class PickActionInform(Action):
    """Picks which information it should give when it predicts the intents
    directive or facilitator.
    It checks which child we are currently interviewing. The index gives us
    the number of the inform utterance, and the index is updated with 1, so that
    the next time this action is called, the child will utter the next piece of
    information.
    If the child has uttered all pieces of information, it will utter_reluctance.
    If the slot child has not been set, it will utter_confusion."""

    def name(self):
        return "action_pick_inform"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],) -> List[Dict[Text, Any]]:

        # imports the child_replies data
        data = pd.read_csv('child_replies.csv')

        # gets the slots index and child
        index = tracker.get_slot('index')
        new_index = index + 1
        child = tracker.get_slot('child')

        # list of number of utter_inform replies for each child
        nr_info_replies = [int(x) for x in list(data.loc[1])[1:]]

        # list of possible child values
        values = list(data.columns)[1:]

        # if slot child is set
        if child != None:
            # finds the correct inform utterance from the data
            i = values.index(child)
            if nr_info_replies[i] >= index:
                message = data[child][index+2]
                dispatcher.utter_message(text=str(message))
                if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                    # imports the hints
                    hints = pd.read_csv('suggestions_replies.csv')
                    hint = list(hints[child])[index+2]
                    dispatcher.utter_message(
                        text='Here comes a hint for what you can say next:')
                    dispatcher.utter_message(text=str(hint))
                return [SlotSet("index", new_index)]
            else:
                # if index has proceeded the number of utter_inform replies for the given child
                if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                    return [FollowupAction(name='utter_reluctance'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_end_conversation')]
                else:
                    return [FollowupAction(name='utter_reluctance')]
        # if the slot child has not been set
        else:
            if next(tracker.get_latest_entity_values('want_hints'), None) is True:
                return [FollowupAction(name='utter_confusion'), FollowupAction(name='utter_heres_a_hint'), FollowupAction(name='utter_hint_invitation')]
            else:
                return [FollowupAction(name='utter_confusion')]
