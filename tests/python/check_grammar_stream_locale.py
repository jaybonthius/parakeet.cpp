#!/usr/bin/env python3
"""Parse actual grammar-stream output independently of the host numeric locale."""
import json
import locale
import os
import pathlib
import shutil
import subprocess
import sys


def reject_constant(value):
    raise ValueError("non-JSON numeric constant: " + value)


def parse_documents(output):
    documents = {"utterances": [], "tail": []}
    for line in output.splitlines():
        label, encoded = line.split("\t", 1)
        document = json.loads(encoded, parse_constant=reject_constant)
        assert set(document) == {"partial", "units", "events"}
        assert isinstance(document["partial"], str)
        for item in document["units"] + document["events"]:
            assert isinstance(item["frame"], int)
            assert type(item["t"]) is float
        for unit in document["units"]:
            assert type(unit["score"]) is float and type(unit["free_score"]) is float
            assert isinstance(unit["frames"], int)
            for alternative in unit["alternatives"]:
                assert isinstance(alternative["text"], str) and type(alternative["score"]) is float
        documents[label].append(document)
    return documents


def check_results(documents):
    expected = "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait"
    keys = ("text", "endpoint_id", "frame", "t")  # grammar scores are checked above
    units = [{key: unit[key] for key in keys}
             for doc in documents["utterances"] for unit in doc["units"]]
    events = [event for doc in documents["utterances"] for event in doc["events"]]
    assert units == [
        {"text": expected, "endpoint_id": 1024, "frame": 98, "t": 7.84},
        {"text": expected, "endpoint_id": 1024, "frame": 194, "t": 15.52},
    ], units
    assert events == [
        {"type": "eou", "id": 1024, "frame": 98, "t": 7.84},
        {"type": "eou", "id": 1024, "frame": 194, "t": 15.52},
    ], events
    assert any(doc["partial"] for doc in documents["utterances"])
    assert any(doc["events"] for doc in documents["utterances"][:-1]), "no live feed event"
    tails = [unit for doc in documents["tail"] for unit in doc["units"]]
    assert len(tails) == 1 and tails[0]["text"] == expected and tails[0]["endpoint_id"] == -1
    assert tails[0]["frame"] == 94 and tails[0]["t"] == 7.52, tails
    assert all(not doc["events"] for doc in documents["tail"]), "fabricated final event"
    assert all(doc["partial"] == "" for doc in (documents["utterances"][-1], documents["tail"][-1]))


def run(executable, name, env):
    result = subprocess.run([executable, name], env=env, text=True, capture_output=True)
    if result.returncode not in (0, 77):
        sys.stderr.write(result.stderr)
        raise RuntimeError("locale stream producer failed: " + str(result.returncode))
    return result


def main():
    executable, locale_dir = sys.argv[1:]
    if not os.environ.get("PARAKEET_TEST_GGUF_EOU"):
        return 77
    env = os.environ.copy()
    name = "de_DE.UTF-8"
    comma = run(executable, name, env)
    if comma.returncode == 77:
        localedef = shutil.which("localedef")
        if not localedef:
            print("skip: comma-decimal locale and localedef unavailable")
            return 77
        directory = pathlib.Path(locale_dir)
        directory.mkdir(parents=True, exist_ok=True)
        generated = subprocess.run([localedef, "--no-archive", "-i", "de_DE", "-f", "UTF-8",
                                    str(directory / name)], text=True, capture_output=True)
        if generated.returncode:
            sys.stderr.write(generated.stderr)
            print("skip: cannot generate checkout-local comma-decimal locale")
            return 77
        env["LOCPATH"] = str(directory) + (os.pathsep + env["LOCPATH"] if env.get("LOCPATH") else "")
        comma = run(executable, name, env)
    assert comma.returncode == 0, "generated locale unavailable to C++ producer"
    reference = run(executable, "C", env)
    assert reference.returncode == 0
    # json.loads has its own JSON number grammar, not locale-sensitive strtod.
    # Check that explicitly under the same comma-decimal C locale as the producer.
    previous_path = os.environ.get("LOCPATH")
    previous_numeric = locale.setlocale(locale.LC_NUMERIC)
    try:
        if "LOCPATH" in env:
            os.environ["LOCPATH"] = env["LOCPATH"]
        locale.setlocale(locale.LC_NUMERIC, name)
        assert locale.localeconv()["decimal_point"] == ","
        assert json.loads('{"t":7.840000}') == {"t": 7.84}
        try:
            json.loads('{"t":7,840000}')
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("test parser accepted invalid comma-decimal JSON")
        parsed = parse_documents(comma.stdout)
        check_results(parsed)
        assert parsed == parse_documents(reference.stdout), "numeric locale changed stream output"
        assert comma.stdout == reference.stdout, "six-decimal wire formatting changed with locale"
    finally:
        locale.setlocale(locale.LC_NUMERIC, previous_numeric)
        if previous_path is None:
            os.environ.pop("LOCPATH", None)
        else:
            os.environ["LOCPATH"] = previous_path
    print("grammar stream locale PASS: every feed/event/final-tail document parsed as standard JSON; C/comma output identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
