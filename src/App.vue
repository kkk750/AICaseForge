<template>
  <div class="app-container">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
      <div class="container">
        <router-link class="navbar-brand" to="/">
          <img src="/static/images/logo-white.png" alt="Logo" height="40">
        </router-link>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
          <ul class="navbar-nav me-auto">
            <li class="nav-item">
              <router-link class="nav-link" to="/">主页</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/workbench">工作台</router-link>
            </li>
            <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown">资源</a>
              <ul class="dropdown-menu dropdown-menu-dark">
                <li><a class="dropdown-item" href="https://github.com/kkk750/AICaseForge" target="_blank">远程仓库</a></li>
                <li><a class="dropdown-item" href="#">Resource Link 2</a></li>
                <li><a class="dropdown-item" href="#">Resource Link 3</a></li>
              </ul>
            </li>
          </ul>
          <div class="navbar-nav">
            <a class="nav-link" href="https://github.com/kkk750/AICaseForge" target="_blank">Docs</a>
          </div>
        </div>
      </div>
    </nav>

    <main>
      <router-view></router-view>
    </main>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>
