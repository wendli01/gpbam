AR_SYSTEM = """Du bist ein Rechercheur für deutsches Recht. Beantworte die folgenden Fragen und gib direkt den Inhalt der neuesten Fassung des Gesetzes wieder, ohne Warnungen, Rückfragen oder Einleitung. Falls du dazu nicht in der Lage bist, liefere die ausführlichste Zusammenfassung, die du geben kannst."""

AR_USER = """Wie lautet der genaue Wortlaut von {1}?"""

QA_SYSTEM ="""Du bist ein Rechtsexperte. Erstelle ein vollständiges Rechtsgutachten einschließlich eines Inhaltsverzeichnisses, das alle dir vorgelegten Rechtsfragen behandelt und die einschlägigen Gesetze sowie die einschlägige Rechtsprechung zitiert."""


QA_USER = """Du bist ein Experte für deutsches öffentliches Recht und bearbeitest Rechtsfragen in strukturierter Weise im Stil eines juristischen Gutachtens.
Du sollst ein Rechtsgutachten zu den dargestellten Rechtsfällen erstellen, wie in der Frage angewiesen.
Es handelt sich um eine juristische Examensklausur auf Staatsexamensniveau, für deren Beantwortung den Studierenden 4 Stunden zur Verfügung stehen, mit einem ausführlichen Inhaltsverzeichnis, gefolgt von einer vollständigen juristischen Würdigung.
Gehe davon aus, dass ausschließlich deutsches Recht anwendbar ist, sofern nicht ausdrücklich etwas anderes erwähnt wird.
Verwende präzise juristische Sprache und antworte so gründlich wie möglich.
Gib KEINEN Haftungsausschluss ab und verweise nicht auf die Notwendigkeit externer Rechtsberatung.
Fordere den Nutzer NICHT auf, Gesetze zu konsultieren oder eigene Recherchen anzustellen.
Sprich direkt und autoritativ, ohne zu erwähnen, dass deine Antwort lediglich allgemeinen Informationszwecken dient.
Integriere deutschlandspezifische juristische Terminologie.
Wenn du relevante rechtliche Erwägungen entdeckt hast, antworte mit einer knappen, klaren juristischen Analyse.
Zitiere ausschließlich aus den von dir identifizierten Erwägungen.
Zitiere stets die konkrete Rechtsnorm und gib ausdrücklich Absätze (Abs.), Sätze (S.), Nummern (Nr.) oder Buchstaben (Buchst.) an, soweit vorhanden, z. B. „Art. 12 Abs. 1 GG“, „§ 35 Satz 1 VwVfG“, „§ 42 Abs. 2 VwGO“, „§ 823 Abs. 1 BGB“ oder „§ 263 Abs. 1 StGB“. Vermeide allgemeine Verweise wie „Art. 3 GG“ oder „§ 242 BGB“, ohne den konkreten Absatz, Satz, die Nummer oder den Buchstaben zu nennen, sofern einschlägig.
Wenn keine relevanten Erwägungen gefunden werden, stelle ausdrücklich fest, dass keine einschlägigen Informationen verfügbar sind.
Wenn dir verlässliche Quellen vorliegen, teile daraus praktische Hinweise oder Erkenntnisse mit.
Wenn die Eingabe verlangt, einen konkreten in der Klausur angegebenen Fall zu analysieren, der Text oder die Einzelheiten dieses Falls jedoch in der Eingabe nicht bereitgestellt wurden, weise ausdrücklich darauf hin, dass das erforderliche Fallmaterial fehlt.

{}

Frage:
```
{}
```

Antwort:"""


# Used instead of QA_USER when an AnswerGenerator has a retriever attached.
# QA_USER drops the retrieved block into a bare slot and never says what it is,
# while carrying three instructions written for a generic retrieval assistant
# ("cite exclusively from what you identified", "state that no relevant
# information is available"). Those are removed here and replaced with guidance
# that fits a Gutachten written over a partial, possibly irrelevant statute set.
QA_USER_RAG = """Du bist ein Experte für deutsches öffentliches Recht und bearbeitest Rechtsfragen in strukturierter Weise im Stil eines juristischen Gutachtens.
Du sollst ein Rechtsgutachten zu den dargestellten Rechtsfällen erstellen, wie in der Frage angewiesen.
Es handelt sich um eine juristische Examensklausur auf Staatsexamensniveau, für deren Beantwortung den Studierenden 4 Stunden zur Verfügung stehen, mit einem ausführlichen Inhaltsverzeichnis, gefolgt von einer vollständigen juristischen Würdigung.
Gehe davon aus, dass ausschließlich deutsches Recht anwendbar ist, sofern nicht ausdrücklich etwas anderes erwähnt wird.
Verwende präzise juristische Sprache und antworte so gründlich wie möglich.
Gib KEINEN Haftungsausschluss ab und verweise nicht auf die Notwendigkeit externer Rechtsberatung.
Fordere den Nutzer NICHT auf, Gesetze zu konsultieren oder eigene Recherchen anzustellen.
Sprich direkt und autoritativ, ohne zu erwähnen, dass deine Antwort lediglich allgemeinen Informationszwecken dient.
Integriere deutschlandspezifische juristische Terminologie.
Wenn du relevante rechtliche Erwägungen entdeckt hast, antworte mit einer knappen, klaren juristischen Analyse.
Zitiere stets die konkrete Rechtsnorm und gib ausdrücklich Absätze (Abs.), Sätze (S.), Nummern (Nr.) oder Buchstaben (Buchst.) an, soweit vorhanden, z. B. „Art. 12 Abs. 1 GG“, „§ 35 Satz 1 VwVfG“, „§ 42 Abs. 2 VwGO“, „§ 823 Abs. 1 BGB“ oder „§ 263 Abs. 1 StGB“. Vermeide allgemeine Verweise wie „Art. 3 GG“ oder „§ 242 BGB“, ohne den konkreten Absatz, Satz, die Nummer oder den Buchstaben zu nennen, sofern einschlägig.
Wenn die Eingabe verlangt, einen konkreten in der Klausur angegebenen Fall zu analysieren, der Text oder die Einzelheiten dieses Falls jedoch in der Eingabe nicht bereitgestellt wurden, weise ausdrücklich darauf hin, dass das erforderliche Fallmaterial fehlt.

Im Abschnitt <context> findest du Normtexte, die durch eine automatische Suche zu diesem Sachverhalt gefunden wurden. Beachte dabei:
- Die Auswahl ist unvollständig und teilweise unpassend. Sie ersetzt nicht deine eigene Kenntnis des deutschen Rechts.
- Insbesondere kann Landesrecht (z. B. PAG, BayBO, BayVwVfG, BV, GO, LStVG) vollständig fehlen, obwohl es einschlägig ist.
- Wenn eine Vorschrift dort nicht auftaucht, heißt das nicht, dass sie nicht existiert oder nicht einschlägig ist. Zitiere die einschlägigen Normen unabhängig davon, ob ihr Wortlaut unten abgedruckt ist.
- Verwende die abgedruckten Normtexte, soweit sie einschlägig sind, und ignoriere die übrigen. Übernimm keine Vorschrift nur deshalb, weil sie unten steht.
- Die Normtexte stehen in <norm>-Elementen mit den Attributen zitat, gebiet und ueberschrift. Zitiere im Gutachten die Norm selbst (etwa „§ 34 BauGB“), nicht das Element oder seine Position.

{}

Frage:
```
{}
```

Antwort:"""


# identifier: qa2
#
# Everything the v1 prompts say is about register and about how to read the
# context: cite the Absatz, no disclaimer, the retrieved norms may be wrong.
# Nothing in them says how a German exam answer is actually built. The judge
# instruction, meanwhile, scores exactly that -- it docks an answer that "die
# Tiefe der Subsumtion aus der Referenzantwort auslässt" and rewards following
# the reference's structure -- and the free judges' prose says the same thing
# unprompted: the recurring complaints are a missing notwendige Beiladung, a
# Klagebefugnis waved through on the Adressatentheorie alone, and answers that
# are "stark verkürzt" with "mangelnde Tiefe". Those are method failures, not
# knowledge failures, and no retriever can fix them.
#
# So this block adds method and completeness only. It deliberately names no
# case-specific law beyond the Bavarian codes the corpus is drawn from, because
# the point is to test whether telling the model *how* to write closes part of
# the 2.75-point gap between the best retrieval arm and the gold-norm oracle --
# not to leak the solutions into the prompt.
#
# Braces are avoided throughout: these strings are .format()ed twice.
QA_METHOD = """Methodische Vorgaben für die Bearbeitung:
- Arbeite durchgehend im Gutachtenstil: Obersatz, Definition, Subsumtion, Ergebnis. Nur bei offensichtlich unproblematischen Punkten ist der Urteilsstil zulässig.
- Prüfe die Zulässigkeit einer verwaltungsgerichtlichen Klage vollständig und in dieser Reihenfolge, jeden Punkt ausdrücklich: Verwaltungsrechtsweg (§ 40 Abs. 1 S. 1 VwGO), statthafte Klageart (§ 88 VwGO), Klagebefugnis (§ 42 Abs. 2 VwGO), Vorverfahren (§§ 68 ff. VwGO) einschließlich etwaiger Entbehrlichkeit, Klagefrist (§ 74 VwGO), Beteiligten- und Prozessfähigkeit (§§ 61, 62 VwGO), notwendige Beiladung (§ 65 Abs. 2 VwGO) sowie das allgemeine Rechtsschutzbedürfnis. Ein unproblematischer Punkt wird kurz festgestellt, aber niemals übergangen.
- Begründe die Klagebefugnis nicht allein mit der Adressatentheorie. Die Adressatenstellung belegt nur die formelle Beschwer; maßgeblich ist die Möglichkeit einer materiellen Rechtsverletzung. Benenne die konkrete Norm, aus der sich das subjektive Recht ergibt.
- Beantworte jede im Sachverhalt gestellte Fallfrage gesondert und in der gestellten Reihenfolge, jeweils mit eigenem Ergebnis. Klagen mehrere Personen oder werden mehrere Bescheide angegriffen, prüfe für jede Person und jeden Bescheid getrennt.
- Stelle bei umstrittenen Fragen die vertretenen Auffassungen mit ihren Argumenten dar und entscheide den Streit, soweit die Auffassungen zu unterschiedlichen Ergebnissen führen. Begründe die Entscheidung.
- Subsumiere am konkreten Sachverhalt: verwende die Tatsachen des Falles ausdrücklich, statt die Voraussetzungen einer Norm abstrakt zu wiederholen.
- Es handelt sich um bayerisches Landesrecht in Verbindung mit Bundesrecht. Prüfe das einschlägige Landesrecht (u. a. BayVwVfG, BayBO, BayVwZVG, PAG, LStVG, GO, BV) ausdrücklich mit; dass sein Wortlaut dir nicht vorliegt, entbindet nicht von seiner Prüfung.
- Kürze nicht: eine knappe Antwort, die eine erforderliche Prüfungsstufe auslässt, ist falsch. Die Bearbeitungszeit von vier Stunden ist auszuschöpfen."""

#: where the method block goes -- immediately before the context slot, so it is
#: the last thing read before the material and the question
_QA_ANCHOR = '\n\n{}\n\nFrage:'


def _with_method(prompt, name):
    if _QA_ANCHOR not in prompt:
        raise AssertionError(f'{name} no longer ends with the context slot; '
                             'QA_METHOD needs re-anchoring')
    return prompt.replace(_QA_ANCHOR, f'\n\n{QA_METHOD}{_QA_ANCHOR}')


QA_USER_V2 = _with_method(QA_USER, 'QA_USER')
QA_USER_RAG_V2 = _with_method(QA_USER_RAG, 'QA_USER_RAG')

#: ``(plain, rag)`` by identifier. v1 must keep producing byte-identical
#: prompts -- every published arm used it.
QA_PROMPTS_BY_NAME = {
    'qa1': (QA_USER, QA_USER_RAG),
    'qa2': (QA_USER_V2, QA_USER_RAG_V2),
}


JUDGE_SYSTEM = """Handle als Prüfer, der auf die Bewertung universitärer Klausuren an deutschen rechtswissenschaftlichen Fakultäten spezialisiert ist.
Deine Aufgabe ist es, zu beurteilen, wie gut die Antwort mit der Musterlösung übereinstimmt, mit Schwerpunkt auf Richtigkeit, Vollständigkeit und juristischer Argumentation."""


# identifier: ji1
JUDGE_INSTRUCTION_V1 = """Ziel:
Deine Aufgabe ist es zu beurteilen, wie gut die Antwort mit der Referenzantwort übereinstimmt, wobei der Schwerpunkt auf Genauigkeit, Vollständigkeit und juristischer Argumentation liegt.
Der generierte Aufsatz muss alle Aspekte des Referenzaufsatzes in ähnlicher Detailtiefe und mit demselben Ergebnis abdecken, um die volle Punktzahl zu erhalten. 
Wenn Punkte, die für die Referenzantwort wesentlich sind, nicht erörtert werden, sollte dies zu einem erheblichen Punktabzug führen, da dadurch die juristische Argumentation unterbrochen wird. 
Der generierte Aufsatz sollte daher einer ähnlichen Struktur wie der Referenzaufsatz folgen.
Eine Antwort, die kurz ist, aber die Tiefe der Subsumtion aus der Referenzantwort auslässt, muss deutlich niedriger bewertet werden als eine lange Antwort, die eine Subsumtion versucht, selbst wenn die lange Antwort geringfügige stilistische Abweichungen enthält.

Kontext:
Dir wird eine vollständige juristische Antwort bereitgestellt (gekennzeichnet als: Antwort des Modells im <antwort_des_modells>-Tag), die auf einer juristischen Examensfrage (gekennzeichnet als: Frage im <frage>-Tag) und einer Musterlösung (gekennzeichnet als: Referenzantwort im <referenzantwort>-Tag) basiert.

Rückgabeformat:
    Nach der Überprüfung der Antwort:
    1. Erklärung: Erkläre deine Begründung dazu, inwiefern der generierte Aufsatz mit dem Referenzaufsatz übereinstimmt oder von ihm abweicht. 
        Identifiziere alle zentralen juristischen Schritte in der Referenzantwort. Enthält die Modellantwort Schritt 1? Schritt 2?
    2. Konstruktives Feedback: Gib zusätzlich neutrales, konstruktives Feedback und Korrekturen im Stil eines Universitätsprofessors.
    3. Korrektheitsbewertung: Vergib eine abschließende Korrektheitsbewertung auf einer Skala von 0,0 bis 1,0 (in Schritten von 0,1). Diese Bewertung sollte widerspiegeln, in welchem Umfang die Antwort die Referenzantwort erfüllt, wobei 
        - 1,0 = vollständige Erfüllung (100 %) 
        - niedrigere Werte verhältnismäßige Defizite widerspiegeln (z. B. 0,5 = 50 % Erfüllung). 
        - Halte dich strikt an das Format: \"[[Punktzahl]]\", z. B. \"Die Richtigkeitsbewertung: [[0,5]]\".


Warnhinweise:
    - Abweichungen oder zusätzliche Elemente, die in der Referenzantwort nicht enthalten sind, sollten unberücksichtigt bleiben, sofern du nicht sicher bist, dass sie juristisch korrekt und relevant sind. 
        Gehe davon aus, dass die Referenzantwort alle Informationen enthält, die für eine perfekte Antwort erforderlich sind.
    - Die Referenzantwort kann Zitate enthalten (z. B. aus Büchern oder juristischen Fachaufsätzen), die die Antwort nicht wiedergeben muss. 
        Gesetzesbestimmungen sollten jedoch präzise zitiert werden, wobei Abs., S., Nr. oder Buchst. anzugeben sind, sofern dies anwendbar ist.
    - Wenn die Referenzantwort separate Unterpunkte enthält, nutze diese als Orientierung für eine proportionale Bewertung (z. B. entspricht die korrekte Behandlung von 2 von 4 Unterpunkten ungefähr einer Bewertung von 0,5)."""


# This judge prompt is taking into consideration of Simon's comment. JUDGE_INSTRUCTION_V1 is the original by lorenz.
# identifier: ji2
JUDGE_INSTRUCTION_V2 = """Ziel:
Deine Aufgabe ist es zu beurteilen, wie gut die Antwort mit der Referenzantwort übereinstimmt, wobei der Schwerpunkt auf Genauigkeit, Vollständigkeit und juristischer Argumentation liegt.
Die generierte Antwort muss alle Aspekte der Referenzantwort in ähnlicher Detailtiefe und mit demselben Ergebnis abdecken, um die volle Punktzahl zu erhalten. 
Wenn Punkte, die für die Referenzantwort wesentlich sind, nicht erörtert werden, sollte dies zu einem erheblichen Punktabzug führen, da dadurch die juristische Argumentation unterbrochen wird.
Die generierte Antwort sollte daher einer ähnlichen Struktur wie die Referenzantwort folgen. Jede Abweichung von der Referenzantwort muss im Hinblick auf die in der Referenzantwort gesetzten Maßstäbe präzise begründet werden.
Eine Antwort, die kurz ist, aber die Tiefe der Subsumtion aus der Referenzantwort auslässt, muss deutlich niedriger bewertet werden als eine lange Antwort, die eine Subsumtion versucht, selbst wenn die lange Antwort geringfügige stilistische Abweichungen enthält.
Daher ist in einem ersten Schritt die Referenzantwort in logisch zusammenhängende Inhaltsabschnitte aufzuteilen, und diese sind den entsprechenden Absätzen in der Antwort des Modells zuzuordnen. Gib für jeden dieser Absätze Feedback gemäß dem unten im Rückgabeformat beschriebenen Schema. Stelle sicher, dass alle Normzitate im selben Inhaltsbereich ebenfalls korrekt zitiert sind.

Kontext:
Dir wird eine vollständige juristische Antwort bereitgestellt (gekennzeichnet als: Antwort des Modells im <antwort_des_modells>-Tag), die auf einer juristischen Examensfrage (gekennzeichnet als: Frage im <frage>-Tag) und einer Musterlösung (gekennzeichnet als: Referenzantwort im <referenzantwort>-Tag) basiert.

Rückgabeformat:
    Nach der Überprüfung der Antwort:
    1. Erklärung: Erkläre deine Begründung dazu, inwiefern der generierte Aufsatz mit dem Referenzaufsatz übereinstimmt oder von ihm abweicht. 
        Identifiziere alle zentralen juristischen Schritte in der Referenzantwort wie oben beschrieben. Enthält die Antwort des Modells Schritt 1? Schritt 2?
    2. Konstruktives Feedback: Gib zusätzlich neutrales, konstruktives Feedback und Korrekturen im Stil eines Universitätsprofessors. Das Feedback sollte präzise und direkt aufzeigen, was falsch ist, und Hinweise zur Verbesserung geben.
    3. Korrektheitsbewertung: Vergib eine abschließende Korrektheitsbewertung auf einer Skala von 0,0 bis 1,0 (in Schritten von 0,1). Diese Bewertung sollte widerspiegeln, in welchem Umfang die Antwort die Referenzantwort erfüllt, wobei 
        - 1,0 = vollständige Erfüllung (100 %) 
        - niedrigere Werte verhältnismäßige Defizite widerspiegeln (z. B. 0,5 = 50 % Erfüllung). 
        - Halte dich strikt an das Format: \"[[Punktzahl]]\", z. B. \"Die Richtigkeitsbewertung: [[0,5]]\".

Warnhinweise:
    - Abweichungen oder zusätzliche Elemente, die in der Referenzantwort nicht enthalten sind, sollten unberücksichtigt bleiben, sofern du nicht sicher bist, dass sie juristisch korrekt und relevant sind. 
        Gehe davon aus, dass die Referenzantwort alle Informationen enthält, die für eine perfekte Antwort und Bewertung erforderlich sind.
    - Die Referenzantwort kann Zitate enthalten (z. B. aus Büchern oder juristischen Fachaufsätzen), die die Antwort nicht wiedergeben muss. 
        Gesetzesbestimmungen sollten jedoch präzise zitiert werden, wobei Abs., S., Nr. oder Buchst. anzugeben sind, sofern dies anwendbar ist.
    - Wenn die Referenzantwort separate Unterpunkte enthält, nutze diese als Orientierung für eine proportionale Bewertung (z. B. entspricht die korrekte Behandlung von 2 von 4 Unterpunkten ungefähr einer Bewertung von 0,5)."""

# This judge prompt is taking into consideration of Simon's comment and MY own understanding and discussion with ChatGPT of the tasks and requirment.
# Notice the creation of rubric comemnt [from discussion with chatGPT :) ]. It is from papers
# 1. https://aclanthology.org/2025.emnlp-industry.136.pdf
# 2. https://aclanthology.org/2025.nllp-1.23.pdf
# and more maybe - ask ChatGPT. 
# identifier: ji3
JUDGE_INSTRUCTION_V3 = """Kontext:
Dir wird eine vollständige juristische Antwort bereitgestellt (gekennzeichnet als: Antwort des Modells im <antwort_des_modells>-Tag), die auf einer juristischen Examensfrage (gekennzeichnet als: Frage im <frage>-Tag) und einer Musterlösung (gekennzeichnet als: Referenzantwort im <referenzantwort>-Tag) basiert.

Ziel:
Deine Aufgabe ist es zu beurteilen, wie gut die Antwort mit der Referenzantwort übereinstimmt, wobei der Schwerpunkt auf Richtigkeit, Vollständigkeit und juristischer Argumentation liegt.
Diese Beurteilung muss auch die korrekte Zitierung der einschlägigen Rechtsnormen und die Einhaltung des Gutachtenstils (Definition - Subsumtion - Ergebnis) berücksichtigen.
Die generierte Antwort muss alle Aspekte der Referenzantwort in vergleichbarer Detailtiefe und mit demselben Ergebnis abdecken, um die volle Punktzahl zu erhalten.
Wenn Punkte, die für die Referenzantwort wesentlich sind, nicht erörtert werden, muss dies zu einem erheblichen Punktabzug führen, da dadurch die juristische Argumentation unterbrochen wird.
Die generierte Antwort sollte daher einer ähnlichen Struktur wie die Referenzantwort folgen. Jede Abweichung von der Referenzantwort muss präzise anhand der in der Referenzantwort gesetzten Maßstäbe begründet werden.
Ein übereinstimmendes Endergebnis allein reicht für eine hohe Bewertung nicht aus; die Antwort muss dieses Ergebnis durch die richtigen Rechtsnormen, die richtige Problemstruktur, Definitionen, Subsumtion und Zwischenergebnisse erreichen.
Deutlich kürzere Antworten müssen sorgfältig auf ausgelassene Probleme, Normzitate, Definitionen und Subsumtion geprüft werden; liegen solche Auslassungen vor, muss die Punktzahl erheblich reduziert werden, selbst wenn das Endergebnis mit der Referenzantwort übereinstimmt.
Eine Antwort, die kurz ist, aber die in der Referenz enthaltene Tiefe der Subsumtion auslässt, muss deutlich niedriger bewertet werden als eine lange Antwort, die die Subsumtion versucht, selbst wenn die lange Antwort geringfügige stilistische Abweichungen enthält.
Wandle daher vor der Vergabe der endgültigen Punktzahl die Referenzantwort in atomare juristische Bewertungspunkte um. Beurteile für jeden Punkt, ob die Antwort des Modells ihn vollständig, teilweise, falsch oder überhaupt nicht abdeckt, und gib an, ob der Punkt zentral oder nachrangig für die juristische Argumentation ist. Verweise jeden Bewertungspunkt nach Möglichkeit auf den entsprechenden Absatz oder die entsprechende Passage in der Antwort des Modells. Die endgültige Punktzahl muss auf dieser punktweisen Beurteilung einschließlich korrekter Normzitate beruhen, nicht auf einem Gesamteindruck oder allein auf dem Endergebnis.


Rückgabeformat:
    Nach der Prüfung der Antwort:
    1. Erläuterung: Erläutere deine Begründung dazu, inwiefern der generierte Aufsatz der Referenzlösung entspricht oder von ihr abweicht.
        - Identifiziere alle zentralen juristischen Prüfungsschritte in der Referenzantwort wie oben beschrieben.
    2. Konstruktives Feedback: Gib zusätzlich neutrales, konstruktives Feedback und Korrekturen im Stil eines Universitätsprofessors. Das Feedback soll präzise und direkt aufzeigen, was falsch ist, und Hinweise zur Verbesserung geben.
    3. Richtigkeitsbewertung: Vergib eine endgültige Richtigkeitsbewertung auf einer Skala von 0.0 bis 1.0 (in Schritten von 0.1). Diese Bewertung soll widerspiegeln, in welchem Umfang die Antwort die Referenzantwort erfüllt, wobei
        - 1.0 = vollständige Erfüllung (100 %) bedeutet, also gleichwertige juristische Qualität und vergleichbare analytische Tiefe zur Referenzantwort in allen wesentlichen Punkten.
        - Niedrigere Punktzahlen Mängel widerspiegeln. Die Punktzahl darf nicht mechanisch durch bloßes Zählen von Prüfungspunkten berechnet werden; fehlende zentrale Probleme, falsche Rechtsnormen, fehlende Definitionen, schwache Subsumtion oder unbelegte Schlussfolgerungen müssen die Punktzahl erheblich reduzieren.
        - Ein übereinstimmendes Endergebnis allein reicht für eine hohe Bewertung nicht aus. Wenn die Antwort das richtige Ergebnis ohne die erforderliche juristische Argumentation, Problemstruktur, Zitate, Definitionen und Subsumtion erreicht, muss die Punktzahl entsprechend reduziert werden.
        - Halte dich strikt an das Format: \"[[Punktzahl]]\", z. B. \"Die Richtigkeitsbewertung: [[0,5]]\".
        
Warnhinweise:
    - Abweichungen oder zusätzliche Elemente, die nicht in der Referenzantwort enthalten sind, sollen unberücksichtigt bleiben, es sei denn, du bist sicher, dass sie juristisch korrekt und relevant sind. Gehe davon aus, dass die Referenzantwort alle Informationen enthält, die für eine perfekte Antwort und Bewertung erforderlich sind.
    - Die Beurteilung muss auch die korrekte Zitierung der einschlägigen Rechtsnormen und die Einhaltung des Gutachtenstils (Definition - Subsumtion - Ergebnis) berücksichtigen.
    - Die Antwort des Modells muss sekundäre Quellenangaben aus der Referenzantwort, wie Bücher, Kommentare oder rechtswissenschaftliche Aufsätze, nicht übernehmen. Gesetzliche Vorschriften müssen jedoch korrekt und präzise zitiert werden, einschließlich Abs., S., Nr. oder Buchst., sofern anwendbar. Wenn die Antwort des Modells eine andere gesetzliche Vorschrift zitiert als die Referenzantwort, soll sie nur dann berücksichtigt werden, wenn die Vorschrift rechtlich gleichwertig, eigenständig korrekt und für dieselbe Rechtsfrage relevant ist; andernfalls muss die Abweichung die Punktzahl reduzieren."""



JUDGE_INSTRUCTIONS_BY_NAME = {
    "ji1": JUDGE_INSTRUCTION_V1,
    "ji2": JUDGE_INSTRUCTION_V2,
    "ji3": JUDGE_INSTRUCTION_V3,
}


def build_judge_user(instruction: str) -> str:
    """Build a JUDGE_USER template around a given judge instruction (e.g. one of the
    JUDGE_INSTRUCTION_V* variants above), so the instruction can be swapped per-run
    without editing this module."""
    return f"""Bewerte den nachstehenden Fall, gib eine kurze Begründung und die endgültige Richtigkeitsbewertung an.

{instruction}

Frage:

<frage>
{{}}
</frage>


Referenzantwort:

<referenzantwort>
{{}}
</referenzantwort>

Antwort des Modells:

<antwort_des_modells>
{{}}
</antwort_des_modells>

Dein Urteil:"""


JUDGE_USER = build_judge_user(JUDGE_INSTRUCTION_V2)