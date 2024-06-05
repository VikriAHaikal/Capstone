const Hapi = require('@hapi/hapi');

(async () => {
  const server = Hapi.server({
    port: 3000,
    host: 'localhost',
    routes: {
      cors: {
        origin: ['*'],
      },
    },
  });

  server.route([
    {
      method: 'POST',
      path: '/signup',
      handler: (request, h) => {
        const { name, email, password } = request.payload;
        const users = []; // temporary db
        const userExists = users.find(user => user.email === email);
        if (userExists) return h.response({ error: 'User already exists.' }).code(400);
        const user = { id: users.length + 1, name, email, password };
        users.push(user);
        return h.response({ result: 'User created successfully.' });
      },
    },
    {
      method: 'POST',
      path: '/login',
      handler: (request, h) => {
        const { email, password } = request.payload;
        const users = []; // temporary db
        const user = users.find(user => user.email === email && user.password === password);
        if (!user) return h.response({ error: 'Invalid email or password.' }).code(401);
        return h.response({ result: 'Logged in successfully.' });
      },
    },
    {
      method: 'GET',
      path: '/logout',
      handler: (request, h) => {
        return h.response({ result: 'Logged out successfully.' });
      },
    },
  ]);

  await server.start();
  console.log(`Server start at: ${server.info.uri}`);
})();