const express = require('express');
const bodyParser = require('body-parser');
const app = express();
app.use(bodyParser.json());

const users = [];

app.post('/signup', (req, res) => {
  const { name, email, password } = req.body;
  
  const userExists = users.find(user => user.email === email);
  
  if (userExists) return res.status(400).json({ error: 'User already exists.' });
  
  const user = { id: users.length + 1, name, email, password };
  users.push(user);
  
  res.json({ result: 'User created successfully.' });
});

app.post('/login', (req, res) => {
  const { email, password } = req.body;
  
  const user = users.find(user => user.email === email && user.password === password);
  
  if (!user) return res.status(401).json({ error: 'Invalid email or password.' });
  
  res.json({ result: 'Logged in successfully.' });
});

app.get('/logout', (req, res) => {
  res.json({ result: 'Logged out successfully.' });
});
