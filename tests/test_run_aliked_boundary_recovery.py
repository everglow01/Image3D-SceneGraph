from pathlib import Path
import runpy
import sqlite3

import pytest


def experiment(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_aliked_boundary_recovery.py"))


def make_database(path):
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE cameras(camera_id,model,width,height,params,prior_focal_length);
            CREATE TABLE images(image_id,name,camera_id);
            CREATE TABLE keypoints(image_id,rows,cols,data);
            CREATE TABLE descriptors(image_id,type,rows,cols,data);
            CREATE TABLE matches(pair_id,rows,cols,data);
            CREATE TABLE two_view_geometries(pair_id,rows,cols,data,config,F,E,H,qvec,tvec);
            INSERT INTO cameras VALUES(1,4,10,20,X'01',0);
            INSERT INTO images VALUES(1,'a.jpg',1),(2,'b.jpg',1),(3,'c.jpg',1);
            INSERT INTO keypoints VALUES(1,1,2,X'01'),(2,1,2,X'02'),(3,1,2,X'03');
            INSERT INTO descriptors VALUES(1,0,1,128,X'04'),(2,0,1,128,X'05'),(3,0,1,128,X'06');
            INSERT INTO matches VALUES(2147483649,0,2,X''),(2147483650,1,2,X'0102');
            INSERT INTO two_view_geometries VALUES(2147483649,0,2,X'',0,NULL,NULL,NULL,NULL,NULL);
            INSERT INTO two_view_geometries VALUES(2147483650,1,2,X'0102',3,X'01',NULL,NULL,NULL,NULL);
        """)


def test_boundary_database_copy_changes_only_target_pairs(tmp_path, monkeypatch):
    module = experiment(monkeypatch)
    source, copied = tmp_path / "source.db", tmp_path / "copied.db"
    make_database(source)
    pair = 2147483649
    contract, before = module["copy_database_for_pairs"](source, copied, [pair])
    assert before == {"candidate": {pair: 0}, "verified": {pair: 0}}
    with sqlite3.connect(source) as old, sqlite3.connect(copied) as new:
        assert module["database_contract"](old, [pair]) == contract
        assert module["database_contract"](new, [pair]) == contract
        assert new.execute("SELECT count(*) FROM matches WHERE pair_id=?", (pair,)).fetchone()[0] == 0
        new.execute("UPDATE keypoints SET data=X'ff' WHERE image_id=1")
        assert module["database_contract"](new, [pair]) != contract
    with pytest.raises(ValueError, match="overwrite"):
        module["copy_database_for_pairs"](source, copied, [pair])


def test_boundary_database_copy_rejects_nonempty_target(tmp_path, monkeypatch):
    module = experiment(monkeypatch)
    source = tmp_path / "source.db"
    make_database(source)
    with pytest.raises(ValueError, match="not empty"):
        module["copy_database_for_pairs"](source, tmp_path / "copy.db", [2147483650])
