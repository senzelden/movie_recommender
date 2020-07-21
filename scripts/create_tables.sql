-- create tables
CREATE TABLE movies (
	movie_id VARCHAR NOT NULL PRIMARY KEY,
	title VARCHAR(300),
	genre VARCHAR(100)
);

CREATE TABLE links (
	movie_id VARCHAR,
	imdbId VARCHAR(20),
	tmdbId VARCHAR(20),
	FOREIGN KEY (movie_id) REFERENCES movies
);

CREATE TABLE ratings (
	user_id VARCHAR,
	movie_id VARCHAR,
	rating REAL,
	time_stamp TIMESTAMP,
	FOREIGN KEY (user_id) REFERENCES movies
);

CREATE TABLE tags (
	user_id VARCHAR,
	movie_id VARCHAR NOT NULL,
	tag VARCHAR(100),
	time_stamp TIMESTAMP
);


-- Inster data into the table
COPY movies FROM '../ml-latest-small/movies.csv' DELIMITER ',' CSV HEADER;
COPY ratings FROM '../data/ml-latest-small/ratings.csv' DELIMITER ',' CSV HEADER;
COPY links FROM '../data/ml-latest-small/links.csv' DELIMITER ',' CSV HEADER;
COPY tags FROM '../data/ml-latest-small/tags.csv' DELIMITER ',' CSV HEADER;
