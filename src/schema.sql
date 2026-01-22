CREATE TABLE IF NOT EXISTS buses (
    "id" INTEGER,
    "license_plate" TEXT NOT NULL UNIQUE,
    "model" TEXT NOT NULL,
    PRIMARY KEY("id")
);

CREATE TABLE IF NOT EXISTS services (
    "id" INTEGER,
    "service_no" INTEGER NOT NULL UNIQUE,
    PRIMARY KEY("id")
);

CREATE TABLE IF NOT EXISTS trips (
    "id" INTEGER,
    "bus_id" INTEGER,
    "service_id" INTEGER,
    "expected" NUMERIC NOT NULL,
    "actual" NUMERIC NOT NULL,
    "residual" NUMERIC NOT NULL,
    "datetime" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY("id"),
    FOREIGN KEY("bus_id") REFERENCES "buses"("id"),
    FOREIGN KEY("service_id") REFERENCES "services"("id")
);

INSERT INTO "buses" ("id", "license_plate", "model")
VALUES
(1, "SBS8413R", "SCANIA_KUB4X2__SD__AC__2_AXLE"),
(2, "SBS8793T", "SCANIA_KUB4X2(EEV)__SD__AC__2"),
(3, "SBS8976H", "SCANIA_KUB4X2(EEV)__SD__AC__2");

INSERT INTO "services" ("id", "service_no")
VALUES
(1, 5);

